"""Slat calibration, target rounding and native tilt compensation regressions."""

from __future__ import annotations

import asyncio
import time
from functools import partial
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError
from OWNd.message import OWNEvent
from voluptuous import Invalid

from custom_components.myhome.const import CONF_COVER_TRAVEL_TIMES
from custom_components.myhome.cover import MyHOMECover, _service_slat_setting
from custom_components.myhome.cover_scope import CoverScope
from custom_components.myhome.cover_slat import (
    SLAT_SETTINGS,
    SlatPlan,
    plan_tilt_compensation,
    snap_tilt_target,
    validate_slat_setting,
)
from custom_components.myhome.cover_strategies import NativePositionStrategy, VenetianTiltStrategy
from custom_components.myhome.validate import cover_schema
from tests.test_cover_calibration import _make_cover


@pytest.mark.parametrize(
    "step,target,expected",
    [
        (1, 12.5, 13),
        (50, 25, 50),
        (50, 24.9, 0),
        (50, 75, 100),
        (100, 49, 0),
        (100, 50, 100),
        (30, 45, 60),
        (30, 76, 90),
        (30, 95, 100),
        (60, 81, 100),
        (60, 79, 60),
        (7, 0, 0),
        (7, 100, 100),
    ],
)
def test_round_targets(step: int, target: float, expected: int) -> None:
    assert snap_tilt_target(target, step) == expected


@pytest.mark.parametrize("key", SLAT_SETTINGS)
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -101, 101, True, None, "bad"])
def test_slat_invalid_values(key: str, bad: object) -> None:
    with pytest.raises(ValueError):
        validate_slat_setting(key, bad)


@pytest.mark.parametrize(
    "key,value",
    [
        ("slat_granularity", 1.5),
        ("slat_time", 0.49),
        ("slat_time_up", 10.01),
        ("slat_time_down", 0),
        ("slat_granularity", 0),
    ],
)
def test_slat_invalid_bounds(key: str, value: float) -> None:
    with pytest.raises(ValueError):
        validate_slat_setting(key, value)


@pytest.mark.parametrize(
    "key,value",
    [
        ("slat_offset_up", 0),
        ("slat_offset", -100),
        ("slat_time", 0.5),
        ("slat_time_down", "10"),
        ("slat_granularity", 100),
    ],
)
def test_slat_valid_bounds(key: str, value: object) -> None:
    assert validate_slat_setting(key, value) == float(value)


def test_yaml_and_service_use_the_same_slat_validation() -> None:
    plain = cover_schema({"blind": {"where": "21", "name": "Blind"}})
    assert "slat_granularity" not in plain["2-21"]
    slat = cover_schema(
        {
            "blind": {
                "where": "21",
                "name": "Blind",
                "slat_tilt": True,
                "slat_offset_up": 0,
            }
        }
    )
    assert slat["2-21"]["slat_granularity"] == 1
    for key in SLAT_SETTINGS:
        with pytest.raises(Invalid):
            cover_schema({"blind": {"where": "21", "name": "Blind", key: 1}})
    for key, bad in (("slat_granularity", 1.5), ("slat_offset", float("nan"))):
        with pytest.raises(Invalid):
            cover_schema(
                {
                    "blind": {
                        "where": "21",
                        "name": "Blind",
                        "slat_tilt": True,
                        key: bad,
                    }
                }
            )
        with pytest.raises(Invalid):
            _service_slat_setting(key, bad)
    assert _service_slat_setting("slat_offset_up", 0) == 0


def _plan(current: int, tilt: int | None, target: int, **kwargs: float) -> SlatPlan:
    params = dict(
        offset_up=1.0,
        offset_down=1.0,
        time_up=2.0,
        time_down=2.0,
        min_delta=1,
        min_pulse=0.4,
    )
    params.update(kwargs)
    return plan_tilt_compensation(current, tilt, target, **params)


@pytest.mark.parametrize(
    "current,tilt,target,pre,move",
    [
        (50, 0, 100, 49, True),
        (50, 100, 0, 51, True),
        (50, 100, 100, 50, False),
        (1, 0, 100, 0, True),
        (99, 100, 0, 100, True),
    ],
)
def test_plan_preserves_current_height(
    current: int,
    tilt: int,
    target: int,
    pre: int,
    move: bool,
) -> None:
    plan = _plan(current, tilt, target)
    assert plan.pre_position == pre and plan.move is move
    assert plan.expected_position == current


def test_fractional_offsets_need_no_height_run() -> None:
    assert not _plan(50, 0, 100, offset_up=0.5).move
    assert not _plan(50, 0, 50, offset_up=-1).move


@pytest.mark.parametrize(
    "current,tilt,target,offset",
    [
        (0, 0, 100, 1),
        (100, 100, 0, 1),
        (50, 0, 100, -1),
    ],
)
def test_infeasible_boundary_or_direction_is_rejected(
    current: int,
    tilt: int,
    target: int,
    offset: int,
) -> None:
    with pytest.raises(ValueError):
        _plan(current, tilt, target, offset_up=offset, offset_down=offset)


def test_plan_never_widens_minimum_movement_or_pulse() -> None:
    with pytest.raises(ValueError, match="minimum height movement"):
        _plan(50, 0, 100, min_delta=2)
    with pytest.raises(ValueError, match="min_motor_pulse"):
        _plan(50, 0, 50, time_up=0.5, time_down=0.5)


def test_directional_timing_and_inheritance() -> None:
    strategy = VenetianTiltStrategy(slat_time=2, slat_time_up=1, slat_time_down=4)
    assert strategy.tilt_pulse_duration_for(0, 50) == 0.5
    assert strategy.tilt_pulse_duration_for(100, 50) == 2
    assert strategy.turn_slats_after(50, 1, "close") == 25
    assert strategy.turn_slats_after(0, 0.25, "open") == 25
    inherited = NativePositionStrategy(slat_time=2)
    inherited.slat_time = 3
    assert inherited.slat_time_for(True) == inherited.slat_time_for(False) == 3
    promoted = strategy.promoted(slat_tilt=True)
    assert promoted.slat_time_for(True) == 1 and promoted.slat_time_for(False) == 4


def _native(hass: HomeAssistant, **kwargs: object) -> tuple[MyHOMECover, MagicMock]:
    gw = MagicMock()
    gw.config_entry.options = {}
    gw.send = AsyncMock(return_value=None)
    gw.send_status_request = AsyncMock()
    c = _make_cover(hass, gw, advanced=True, slat_tilt=True, min_position_delta=1, **kwargs)
    c._dim10_seen = True
    c._attr_current_cover_position = 50
    c._attr_current_cover_tilt_position = 0
    return c, gw


async def _finish_tilt(cover: MyHOMECover) -> None:
    """Deliver the pulse's scheduled stop without a wall-clock delay."""
    assert cover._is_tilting and cover._tilt_start_time is not None
    await cover._async_stop(at=cover._tilt_start_time + cover._tilt_duration)


def _confirm_motion(
    cover: MyHOMECover,
    gateway: MagicMock,
    final: int,
    *,
    height_seconds: float = 2,
    pre_direction_echo: bool = False,
) -> None:
    """Supply actuator reports and real delivery futures for a compensated tilt."""

    async def send(command: object) -> asyncio.Future[float]:
        if cover._compensation_phase == "height" and str(command) != "*2*0*21##":
            assert cover._compensation_plan is not None
            pre = cover._compensation_plan.pre_position
            current = cover.current_cover_position
            assert current is not None
            direction = 1 if pre > current else 2
            expected = f"*2*{direction}*21##" if pre in (0, 100) else f"*#2*21*#11#001*{pre}##"
            assert str(command) == expected
            if pre_direction_echo and pre in (0, 100):
                cover.handle_event(OWNEvent.parse("*2*1000#0*21##"))
                cover.handle_event(OWNEvent.parse(f"*#2*21*10*10*{current}*001*0##"))
                cover.handle_event(OWNEvent.parse("*2*0*21##"))
            started = time.monotonic()
            cover.handle_event(OWNEvent.parse(f"*#2*21*10*{direction + 10}*{current}*001*0##"))
            with patch(
                "custom_components.myhome.cover.time.monotonic",
                return_value=started + height_seconds,
            ):
                cover.handle_event(OWNEvent.parse(f"*#2*21*10*10*{pre}*001*0##"))
        elif str(command) in ("*2*1*21##", "*2*2*21##"):
            direction = 1 if str(command) == "*2*1*21##" else 2
            if pre_direction_echo:
                cover.handle_event(OWNEvent.parse("*2*1000#0*21##"))
                current = cover.current_cover_position
                cover.handle_event(OWNEvent.parse(f"*#2*21*10*10*{current}*001*0##"))
                cover.handle_event(OWNEvent.parse(f"*2*1000#{direction}*21##"))
            cover.handle_event(OWNEvent.parse(str(command)))
        written = asyncio.get_running_loop().create_future()
        written.set_result(time.monotonic())
        return written

    async def status(command: object) -> None:
        assert not cover._is_tilting
        cover.handle_event(OWNEvent.parse(f"*#2*21*10*10*{final}*001*0##"))

    gateway.send.side_effect = send
    gateway.send_status_request.side_effect = status
    cover._async_wait_pulse_end = AsyncMock(side_effect=partial(_finish_tilt, cover))


async def test_persistent_slat_overrides_inherit_only_when_unset(hass: HomeAssistant) -> None:
    c, gw = _native(hass)
    c._persist_calibration = MagicMock()
    await c.async_set_travel_time(
        slat_time_up=1, slat_offset=2, slat_offset_up=0, slat_granularity=50
    )
    stored = c._persist_calibration.call_args.args[0]
    assert stored == {
        "slat_time_up": 1,
        "slat_offset": 2,
        "slat_offset_up": 0,
        "slat_granularity": 50,
    }
    assert c._slat_offset_for(True) == 0 and c._slat_offset_for(False) == 2
    restored, _ = _native(hass, calibration=stored)
    await restored.async_set_travel_time(slat_time=3)
    assert restored._slat_time_for(True) == 1 and restored._slat_time_for(False) == 3
    with pytest.raises(ServiceValidationError):
        await c.async_set_travel_time(slat_offset=4, slat_granularity=1.5)
    assert c._slat_offset == 2
    c._device_config = MagicMock(
        return_value={"slat_time": 2, "slat_offset": -1, "slat_granularity": 7}
    )
    gw.config_entry.options = {CONF_COVER_TRAVEL_TIMES: {"21": stored}}
    with patch.object(hass.config_entries, "async_update_entry"):
        await c.async_reset_travel_time()
    assert c._slat_time_for(True) == c._slat_time_for(False) == 2
    assert c._slat_offset_for(True) == c._slat_offset_for(False) == -1
    assert c._slat_granularity == 7


async def test_height_calibration_preserves_all_slat_settings(hass: HomeAssistant) -> None:
    stored = dict(zip(SLAT_SETTINGS, (3, 1, 4, 2, 0, -1, 7), strict=True))
    gw = MagicMock()
    gw.config_entry.options = {CONF_COVER_TRAVEL_TIMES: {"21": stored}}
    c = _make_cover(hass, gw, slat_tilt=True, calibration=stored)
    c._calibration_run = AsyncMock(side_effect=[20, 22, 24])
    c._persist_calibration = MagicMock()
    with patch("custom_components.myhome.cover.CALIBRATION_SETTLE", 0):
        await c.async_calibrate()
    record = c._persist_calibration.call_args.args[0]
    assert {key: record[key] for key in SLAT_SETTINGS} == stored
    assert record["down"] == 22 and record["up"] == 24
    restored = _make_cover(hass, gw, slat_tilt=True, calibration=record)
    assert restored._slat_time_for(True) == 1 and restored._slat_time_for(False) == 4
    assert restored._slat_offset_for(True) == 0 and restored._slat_offset_for(False) == -1
    assert restored._slat_granularity == 7
    await c.async_set_travel_time(travel_time=20, slat_offset=3)
    manual = c._persist_calibration.call_args.args[0]
    assert manual["slat_time_up"] == 1 and manual["slat_offset_up"] == 0
    assert manual["slat_offset"] == 3 and manual["down"] == 20


async def test_reset_removes_legacy_calibration_key(hass: HomeAssistant) -> None:
    c, gw = _native(hass, device_id="04", where="04")
    gw.config_entry.options = {CONF_COVER_TRAVEL_TIMES: {"0004": {"slat_offset": 2}}}
    with patch.object(hass.config_entries, "async_update_entry") as update:
        await c.async_reset_travel_time()
    assert update.call_args.kwargs["options"][CONF_COVER_TRAVEL_TIMES] == {}


@pytest.mark.parametrize("kind", ["native_zero_offset", "timed", "scope"])
async def test_legacy_tilt_path_is_unchanged(hass: HomeAssistant, kind: str) -> None:
    if kind == "timed":
        c = _make_cover(hass, MagicMock(), slat_tilt=True, slat_offset=1)
        c._attr_current_cover_tilt_position = 0
    else:
        c, _ = _native(hass, slat_offset=0 if kind == "native_zero_offset" else 1)
        if kind == "scope":
            c.scope = CoverScope.of("0")
    c._async_start_tilt = AsyncMock()
    c._async_set_compensated_tilt = AsyncMock()
    await c.async_set_cover_tilt_position(tilt_position=100)
    c._async_start_tilt.assert_awaited_once_with(100)
    c._async_set_compensated_tilt.assert_not_called()


async def test_granularity_rounds_requests_not_estimates(hass: HomeAssistant) -> None:
    c, _ = _native(hass, slat_granularity=50)
    c._attr_current_cover_tilt_position = 37
    c._async_start_tilt = AsyncMock()
    await c.async_set_cover_tilt_position(tilt_position=76)
    c._async_start_tilt.assert_awaited_once_with(100)
    assert c.current_cover_tilt_position == 37


@pytest.mark.parametrize("tilt,target,pre", [(0, 100, 49), (100, 0, 51)])
async def test_public_tilt_compensates_without_changing_requested_height(
    hass: HomeAssistant,
    tilt: int,
    target: int,
    pre: int,
) -> None:
    c, gw = _native(hass, slat_offset=1)
    c._attr_current_cover_tilt_position = tilt
    _confirm_motion(c, gw, 50)
    assert await c.async_set_cover_tilt_position(tilt_position=target) is None
    result = c.extra_state_attributes["slat_compensation"]
    assert result["status"] == "done" and result["pre_position"] == pre
    assert result["current_position"] == 50 and result["current_tilt_position"] == target
    assert result["position_error"] == 0
    assert gw.send.call_count == 3


@pytest.mark.parametrize(
    "current,tilt,target,offset,final,pre_direction_echo",
    [
        (50, 0, 50, 2, 50, False),
        (50, 100, 50, 2, 50, False),
        (50, 0, 50, -1, 49, False),
        (50, 100, 50, -1, 51, False),
        (1, 0, 100, 1, 1, False),
        (99, 100, 0, 1, 99, False),
        (1, 0, 100, 1, 1, True),
        (99, 100, 0, 1, 99, True),
    ],
)
async def test_final_height_report_does_not_overwrite_completed_tilt(
    hass: HomeAssistant,
    current: int,
    tilt: int,
    target: int,
    offset: int,
    final: int,
    pre_direction_echo: bool,
) -> None:
    c, gw = _native(hass, slat_offset=offset)
    c._attr_current_cover_position = current
    c._attr_current_cover_tilt_position = tilt
    _confirm_motion(c, gw, final, pre_direction_echo=pre_direction_echo)
    await c.async_set_cover_tilt_position(tilt_position=target)
    result = c.extra_state_attributes["slat_compensation"]
    assert result["status"] == "done" and result["current_tilt_position"] == target
    assert result["current_position"] == final
    assert result["position_error"] == final - current


@pytest.mark.parametrize(
    "case",
    [
        "height_unknown",
        "no_dim10",
        "moving",
        "calibrating",
        "deadband",
        "floor",
        "boundary",
    ],
)
async def test_infeasible_compensation_sends_nothing(hass: HomeAssistant, case: str) -> None:
    c, gw = _native(hass, slat_offset=1)
    target = 100
    if case == "height_unknown":
        c._attr_current_cover_position = None
    elif case == "no_dim10":
        c._dim10_seen = False
    elif case == "moving":
        c._attr_is_opening = True
    elif case == "calibrating":
        c._calibrating = True
    elif case == "deadband":
        c._strategy.min_position_delta = 2
    elif case == "floor":
        c._slat_time = 0.5
        target = 50
    else:
        c._attr_current_cover_position = 0
    with pytest.raises(HomeAssistantError):
        await c.async_set_cover_tilt_position(tilt_position=target)
    gw.send.assert_not_called()


@pytest.mark.parametrize("tilt", [None, 90])
async def test_remaining_tilt_is_revalidated_after_height_stop(
    hass: HomeAssistant,
    tilt: int | None,
) -> None:
    c, gw = _native(hass, slat_offset=1)
    _confirm_motion(c, gw, 50)

    async def confirmed(event: asyncio.Event, cancel: asyncio.Event, timeout: float) -> None:
        c._attr_current_cover_tilt_position = tilt

    c._wait_compensation_event = AsyncMock(side_effect=confirmed)
    c._async_start_tilt = AsyncMock()
    with pytest.raises(HomeAssistantError, match="unknown|below min_motor_pulse"):
        await c.async_set_cover_tilt_position(tilt_position=100)
    c._async_start_tilt.assert_not_called()


@pytest.mark.parametrize(
    "action", ["stop", "height", "manual_stop", "bus_level", "reversal", "scope_stop", "cancel"]
)
async def test_interruption_prevents_late_tilt(hass: HomeAssistant, action: str) -> None:
    from custom_components.myhome.cover_scope import CoverFamily, MyHOMEScopeCover

    c, gw = _native(hass, slat_offset=1)
    c._async_start_tilt = AsyncMock()
    task = asyncio.create_task(c.async_set_cover_tilt_position(tilt_position=100))
    for _ in range(5):
        await asyncio.sleep(0)
    if action == "stop":
        await c.async_stop_cover()
    elif action == "height":
        await c.async_set_cover_position(position=70)
    elif action == "scope_stop":
        family = CoverFamily()
        family.add(c)
        scope = MyHOMEScopeCover(
            hass,
            name="All",
            entity_name=None,
            device_id="0",
            who="2",
            where="0",
            interface=None,
            advanced=False,
            manufacturer="BTicino",
            model="Shutter",
            gateway=gw,
            scope=CoverScope.of("0"),
        )
        scope._family = family
        await scope.async_stop_cover()
    elif action == "cancel":
        task.cancel()
    else:
        frame = {
            "manual_stop": "*2*1000#0*21##",
            "bus_level": "*#2*21*#11#001#1*70##",
            "reversal": "*2*1*21##",
        }[action]
        c.handle_event(OWNEvent.parse(frame))
    c.handle_event(OWNEvent.parse("*#2*21*10*10*49*001*0##"))
    with pytest.raises((HomeAssistantError, asyncio.CancelledError)):
        await asyncio.wait_for(task, 1)
    c._async_start_tilt.assert_not_called()


@pytest.mark.parametrize("phase", ["height", "tilt"])
@pytest.mark.parametrize("action", ["timeout", "cancel", "stop"])
async def test_abandoned_delivery_is_withdrawn(
    hass: HomeAssistant,
    phase: str,
    action: str,
) -> None:
    c, gw = _native(hass, slat_offset=1)
    target = 100 if phase == "height" else 50
    written = asyncio.get_running_loop().create_future()

    async def send(command: object) -> asyncio.Future[float]:
        if str(command) == "*2*0*21##":
            stopped = asyncio.get_running_loop().create_future()
            stopped.set_result(time.monotonic())
            return stopped
        return written

    gw.send.side_effect = send
    with patch("custom_components.myhome.cover.WRITE_TIMEOUT", 0.01):
        task = asyncio.create_task(c.async_set_cover_tilt_position(tilt_position=target))
        for _ in range(5):
            await asyncio.sleep(0)
        if action == "cancel":
            task.cancel()
        elif action == "stop":
            await c.async_stop_cover()
        with pytest.raises((HomeAssistantError, asyncio.CancelledError)):
            await asyncio.wait_for(task, 1)
    assert written.cancelled() and not c._is_tilting
    assert c.extra_state_attributes["slat_compensation"]["status"] == "interrupted"


async def test_cancelled_height_delivery_is_a_service_error(hass: HomeAssistant) -> None:
    c, gw = _native(hass, slat_offset=1)
    written = asyncio.get_running_loop().create_future()
    written.cancel()
    gw.send.return_value = written
    with pytest.raises(HomeAssistantError, match="cancelled before reaching the bus"):
        await c.async_set_cover_tilt_position(tilt_position=100)


@pytest.mark.parametrize("takeover", ["stop", "height", "scope_tilt"])
async def test_takeover_during_enqueue_cannot_resurrect_old_pulse(
    hass: HomeAssistant,
    takeover: str,
) -> None:
    c, gw = _native(hass, slat_offset=1)
    enqueued, release = asyncio.Event(), asyncio.Event()
    old_written = asyncio.get_running_loop().create_future()

    async def send(command: object) -> asyncio.Future[float]:
        if str(command) == "*2*1*21##":
            enqueued.set()
            await release.wait()
            return old_written
        if str(command) == "*#2*21*#11#001*70##":
            c.handle_event(OWNEvent.parse("*#2*21*10*11*50*001*0##"))
        written = asyncio.get_running_loop().create_future()
        written.set_result(time.monotonic())
        return written

    gw.send.side_effect = send
    task = asyncio.create_task(c.async_set_cover_tilt_position(tilt_position=50))
    await asyncio.wait_for(enqueued.wait(), 1)
    if takeover == "stop":
        await c.async_stop_cover()
    elif takeover == "height":
        await c.async_set_cover_position(position=70)
    else:
        leader, _ = _native(hass, device_id="0", where="0")
        c._cancel_compensation_request()
        c._follow_pulse(leader, "open")
    generation = c._run_generation
    release.set()
    with pytest.raises(HomeAssistantError):
        await asyncio.wait_for(task, 1)
    assert old_written.cancelled() and c._run_generation == generation
    assert c._stop_task is None
    if takeover == "scope_tilt":
        assert c._is_tilting and c._tilt_follows is leader
        assert c._tilt_direction == "open" and c._tilt_target_position == 100
    else:
        assert not c._is_tilting
        assert c.is_opening is (takeover == "height")


@pytest.mark.parametrize("manual_stop", [False, True])
async def test_own_pre_direction_stop_is_not_manual_takeover(
    hass: HomeAssistant,
    manual_stop: bool,
) -> None:
    c, gw = _native(hass, slat_offset=1)
    _confirm_motion(c, gw, 50, pre_direction_echo=True)
    if manual_stop:

        async def stop_after_start() -> None:
            c.handle_event(OWNEvent.parse("*2*1000#0*21##"))

        c._async_wait_pulse_end = AsyncMock(side_effect=stop_after_start)
        with pytest.raises(HomeAssistantError, match="interrupted"):
            await c.async_set_cover_tilt_position(tilt_position=50)
        gw.send_status_request.assert_not_called()
    else:
        await c.async_set_cover_tilt_position(tilt_position=50)
        assert c.extra_state_attributes["slat_compensation"]["status"] == "done"


@pytest.mark.parametrize(
    "feedback", ["same_direction", "reversal", "manual_command", "outside_offset"]
)
async def test_final_confirmation_distinguishes_delayed_feedback_from_takeover(
    hass: HomeAssistant,
    feedback: str,
) -> None:
    c, gw = _native(hass, slat_offset=1)
    _confirm_motion(c, gw, 50)

    async def status(command: object) -> None:
        assert not c._is_tilting and c._compensation_expected_stop
        if feedback == "manual_command":
            c.handle_event(OWNEvent.parse("*2*1000#1*21##"))
        moving = 12 if feedback == "reversal" else 11
        c.handle_event(OWNEvent.parse(f"*#2*21*10*{moving}*50*001*0##"))
        final = 60 if feedback == "outside_offset" else 50
        c.handle_event(OWNEvent.parse(f"*#2*21*10*10*{final}*001*0##"))

    gw.send_status_request.side_effect = status
    if feedback == "same_direction":
        await c.async_set_cover_tilt_position(tilt_position=50)
        result = c.extra_state_attributes["slat_compensation"]
        assert result["status"] == "done" and result["current_tilt_position"] == 50
    else:
        with pytest.raises(HomeAssistantError):
            await c.async_set_cover_tilt_position(tilt_position=50)
        assert c.extra_state_attributes["slat_compensation"]["status"] == "interrupted"


async def test_short_height_run_refreshes_prediction_without_another_run(
    hass: HomeAssistant,
) -> None:
    c, gw = _native(hass, slat_offset=2)
    _confirm_motion(c, gw, 52, height_seconds=0.5)
    await c.async_set_cover_tilt_position(tilt_position=50)
    result = c.extra_state_attributes["slat_compensation"]
    assert result["pre_position"] == 51 and result["tilt_start_position"] == 25
    assert result["height_change"] == 0.5
    assert result["expected_position"] == 51.5 and result["expected_position_error"] == 1.5
    assert result["position_error"] == 2  # Reported height is never corrected by the integration.
    assert gw.send.call_count == 3


async def test_endpoint_echo_window_is_anchored_to_actual_delivery(hass: HomeAssistant) -> None:
    from custom_components.myhome.cover import ECHO_WINDOW

    c, gw = _native(hass, slat_offset=1)
    c._attr_current_cover_position = 1
    written = asyncio.get_running_loop().create_future()
    gw.send.return_value = written
    task = asyncio.create_task(c.async_set_cover_tilt_position(tilt_position=100))
    for _ in range(5):
        await asyncio.sleep(0)
    assert c._pending_cmd == "close"
    delivered_at = time.monotonic()
    written.set_result(delivered_at)
    for _ in range(5):
        await asyncio.sleep(0)
    assert c._echo_until == delivered_at + ECHO_WINDOW
    assert not c._motor_started.is_set()
    _confirm_motion(c, gw, 1)
    c.handle_event(OWNEvent.parse("*#2*21*10*10*0*001*0##"))
    await asyncio.wait_for(task, 1)
    assert c.extra_state_attributes["slat_compensation"]["status"] == "done"


async def test_stop_at_confirmed_height_boundary_prevents_tilt(hass: HomeAssistant) -> None:
    c, gw = _native(hass, slat_offset=1)
    _confirm_motion(c, gw, 50)
    wait = c._wait_compensation_event

    async def stop_after_confirmation(
        event: asyncio.Event | asyncio.Future[float], cancel: asyncio.Event, timeout: float,
    ) -> None:
        await wait(event, cancel, timeout)
        if event is c._compensation_height_stopped and c._compensation_phase == "height":
            await c.async_stop_cover()

    c._wait_compensation_event = AsyncMock(side_effect=stop_after_confirmation)
    c._async_start_tilt = AsyncMock()
    with pytest.raises(HomeAssistantError, match="interrupted"):
        await c.async_set_cover_tilt_position(tilt_position=100)
    c._async_start_tilt.assert_not_called()
    assert c.extra_state_attributes["slat_compensation"]["status"] == "interrupted"


async def test_stop_while_waiting_for_tilt_lock_sends_no_pulse(hass: HomeAssistant) -> None:
    c, gw = _native(hass, slat_offset=0.5)
    c._async_start_tilt = AsyncMock()
    await c._tilt_lock.acquire()
    task = asyncio.create_task(c.async_set_cover_tilt_position(tilt_position=100))
    try:
        for _ in range(5):
            await asyncio.sleep(0)
        assert c._compensation_phase == "tilt"
        await c.async_stop_cover()
    finally:
        c._tilt_lock.release()
    with pytest.raises(HomeAssistantError, match="interrupted"):
        await asyncio.wait_for(task, 1)
    c._async_start_tilt.assert_not_called()
    assert [str(call.args[0]) for call in gw.send.call_args_list] == ["*2*0*21##"]


async def test_gateway_failure_is_visible_in_compensation_result(hass: HomeAssistant) -> None:
    c, gw = _native(hass, slat_offset=1)
    gw.send.side_effect = RuntimeError("gateway disconnected")
    with pytest.raises(HomeAssistantError, match="gateway disconnected"):
        await c.async_set_cover_tilt_position(tilt_position=100)
    result = c.extra_state_attributes["slat_compensation"]
    assert result["status"] == "interrupted" and result["error"] == "gateway disconnected"
    assert c._compensation_cancel is None and not c._is_tilting


@pytest.mark.parametrize("stop", ["same_height", "different_height", "plain_stop"])
async def test_start_height_poll_is_harmless_but_unplanned_stop_aborts(
    hass: HomeAssistant, stop: str,
) -> None:
    c, _ = _native(hass, slat_offset=1)
    c._async_start_tilt = AsyncMock()
    task = asyncio.create_task(c.async_set_cover_tilt_position(tilt_position=100))
    for _ in range(5):
        await asyncio.sleep(0)
    c.handle_event(OWNEvent.parse("*#2*21*10*10*50*001*0##"))
    assert c._compensation_cancel is not None and not c._compensation_cancel.is_set()
    if stop == "same_height":
        c.handle_event(OWNEvent.parse("*#2*21*10*12*50*001*0##"))
        frame = "*#2*21*10*10*50*001*0##"
    elif stop == "different_height":
        frame = "*#2*21*10*10*75*001*0##"
    else:
        frame = "*2*0*21##"
    c.handle_event(OWNEvent.parse(frame))
    with pytest.raises(HomeAssistantError, match="interrupted"):
        await asyncio.wait_for(task, 1)
    c._async_start_tilt.assert_not_called()


async def test_compensated_tilt_uses_real_auto_stop(hass: HomeAssistant) -> None:
    c, gw = _native(hass, slat_offset=0.5, slat_time=0.5)

    async def send(command: object) -> asyncio.Future[float]:
        if str(command) == "*2*1*21##":
            c.handle_event(OWNEvent.parse(str(command)))
        written = asyncio.get_running_loop().create_future()
        written.set_result(time.monotonic())
        return written

    async def status(command: object) -> None:
        c.handle_event(OWNEvent.parse("*#2*21*10*10*50*001*0##"))

    gw.send.side_effect = send
    gw.send_status_request.side_effect = status
    await c.async_set_cover_tilt_position(tilt_position=100)
    result = c.extra_state_attributes["slat_compensation"]
    assert result["status"] == "done" and result["current_tilt_position"] == 100
    assert [str(call.args[0]) for call in gw.send.call_args_list] == ["*2*1*21##", "*2*0*21##"]
    assert not c._is_tilting


async def test_tilt_poll_without_closed_metadata_preserves_pulse(hass: HomeAssistant) -> None:
    c, gw = _native(hass, slat_offset=0.5)
    _confirm_motion(c, gw, 50)
    send = gw.send.side_effect

    async def send_with_poll(command: object) -> asyncio.Future[float]:
        written = await send(command)
        if str(command) == "*2*1*21##":
            started = c._tilt_start_time
            message = OWNEvent.parse("*#2*21*10*10*50*001*0##")
            message._is_closed = None
            c.handle_event(message)
            assert c._is_tilting and c._tilt_start_time == started
            assert c._attr_is_closed is False
        return written

    gw.send.side_effect = send_with_poll
    await c.async_set_cover_tilt_position(tilt_position=100)
    assert c.extra_state_attributes["slat_compensation"]["status"] == "done"


async def test_height_run_establishes_unknown_slat_angle(hass: HomeAssistant) -> None:
    c, gw = _native(hass, slat_offset=1)
    c._attr_current_cover_tilt_position = None
    _confirm_motion(c, gw, 50)
    await c.async_set_cover_tilt_position(tilt_position=100)
    result = c.extra_state_attributes["slat_compensation"]
    assert result["pre_position"] == 49 and result["tilt_start_position"] == 0
    assert result["status"] == "done" and result["current_tilt_position"] == 100


async def test_cancelling_caller_after_motor_start_keeps_real_auto_stop(hass: HomeAssistant) -> None:
    c, gw = _native(hass, slat_offset=0.5, slat_time=0.5)
    written = asyncio.get_running_loop().create_future()
    stopped = asyncio.Event()

    async def send(command: object) -> asyncio.Future[float]:
        if str(command) == "*2*1*21##":
            return written  # The gateway is still awaiting its command-session reply.
        delivered = asyncio.get_running_loop().create_future()
        delivered.set_result(time.monotonic())
        if str(command) == "*2*0*21##":
            stopped.set()
        return delivered

    gw.send.side_effect = send
    task = asyncio.create_task(c.async_set_cover_tilt_position(tilt_position=100))
    for _ in range(5):
        await asyncio.sleep(0)
    assert c._pulse_written is written and c._stop_task is not None
    written_at = time.monotonic()
    c.handle_event(OWNEvent.parse("*2*1*21##"))
    assert c._motor_started.is_set() and c._is_tilting
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not written.cancelled()
    assert c._is_tilting and c._stop_task is not None
    written.set_result(written_at)
    await asyncio.wait_for(stopped.wait(), 2)
    assert [str(call.args[0]) for call in gw.send.call_args_list] == ["*2*1*21##", "*2*0*21##"]
    assert not c._is_tilting and c.current_cover_tilt_position == 100


async def test_standard_ha_tilt_service_dispatches_to_compensation(hass: HomeAssistant) -> None:
    import logging
    from datetime import timedelta

    from homeassistant.helpers.entity_platform import EntityPlatform
    from homeassistant.setup import async_setup_component
    from pytest_homeassistant_custom_component.common import MockConfigEntry

    from custom_components.myhome import cover as cover_platform
    from custom_components.myhome.const import DOMAIN
    from tests.conftest import attach_runtime

    mac = "00:03:50:00:00:01"
    entry = MockConfigEntry(domain=DOMAIN, data={"mac": mac})
    entry.add_to_hass(hass)
    gateway = MagicMock(mac=mac, available=True, device_registry_id=None)
    gateway.is_who_available.return_value = True
    gateway.send = AsyncMock()
    gateway.send_status_request = AsyncMock()
    runtime = attach_runtime(hass, entry, mac, gateway)
    runtime.platforms["cover"] = {"21": {
        "where": "21", "name": "Blind", "advanced": True,
        "slat_tilt": True, "slat_offset": 1, "slat_granularity": 50,
    }}
    assert await async_setup_component(hass, "cover", {})
    platform = EntityPlatform(
        hass=hass, logger=logging.getLogger(__name__), domain="cover", platform_name=DOMAIN,
        platform=cover_platform, scan_interval=timedelta(seconds=30), entity_namespace=None,
    )
    assert await platform.async_setup_entry(entry)
    try:
        cover = next(iter(platform.entities.values()))
        cover._async_set_compensated_tilt = AsyncMock()
        cover.handle_event(OWNEvent.parse("*#2*21*10*10*50*001*0##"))
        assert hass.states.get(cover.entity_id) is not None
        await hass.services.async_call(
            "cover", "set_cover_tilt_position",
            {"entity_id": cover.entity_id, "tilt_position": 76}, blocking=True,
        )
        cover._async_set_compensated_tilt.assert_awaited_once_with(100)
        gateway.send.assert_not_called()
    finally:
        await platform.async_reset()
