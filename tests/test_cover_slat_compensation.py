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
    kwargs.setdefault("min_position_delta", 1)
    c = _make_cover(hass, gw, advanced=True, slat_tilt=True, **kwargs)
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


@pytest.mark.parametrize("missing", ["height", "dim10"])
@pytest.mark.parametrize("opening", [True, False])
@pytest.mark.parametrize("unknown_tilt", [False, True])
async def test_zero_direction_offset_uses_legacy_without_native_feedback(
    hass: HomeAssistant, missing: str, opening: bool, unknown_tilt: bool,
) -> None:
    c, gw = _native(
        hass, slat_offset=10,
        slat_offset_up=0 if opening else None,
        slat_offset_down=None if opening else 0,
    )
    c._attr_current_cover_tilt_position = None if unknown_tilt else (0 if opening else 100)
    if missing == "height":
        c._attr_current_cover_position = None
    else:
        c._dim10_seen = False
    c._async_start_tilt = AsyncMock()
    c._async_set_compensated_tilt = AsyncMock()
    target = 100 if opening else 0
    await c.async_set_cover_tilt_position(tilt_position=target)
    c._async_start_tilt.assert_awaited_once_with(target)
    c._async_set_compensated_tilt.assert_not_called()
    gw.send.assert_not_called()


@pytest.mark.parametrize("offset_up,offset_down", [(2, 0), (0, 2)])
async def test_unknown_partial_tilt_uses_compensation_for_either_offset(
    hass: HomeAssistant, offset_up: float, offset_down: float,
) -> None:
    c, _ = _native(hass, slat_offset_up=offset_up, slat_offset_down=offset_down)
    c._attr_current_cover_tilt_position = None
    c._async_start_tilt = AsyncMock()
    c._async_set_compensated_tilt = AsyncMock()
    await c.async_set_cover_tilt_position(tilt_position=50)
    c._async_set_compensated_tilt.assert_awaited_once_with(50)
    c._async_start_tilt.assert_not_called()


@pytest.mark.parametrize("opening_pulse", [True, False])
async def test_direction_routing_uses_running_estimate_not_stale_tilt(
    hass: HomeAssistant, opening_pulse: bool,
) -> None:
    c, _ = _native(
        hass, slat_offset_up=0 if opening_pulse else 2,
        slat_offset_down=2 if opening_pulse else 0,
    )
    c._is_tilting = True
    c._tilt_initial_position = c._attr_current_cover_tilt_position = 0 if opening_pulse else 100
    c._tilt_target_position = 100 if opening_pulse else 0
    c._tilt_direction = "open" if opening_pulse else "close"
    c._tilt_start_time = 10
    c._async_start_tilt = AsyncMock()
    c._async_set_compensated_tilt = AsyncMock()
    target = 60 if opening_pulse else 40
    with patch("custom_components.myhome.cover.time.monotonic", return_value=11.4):
        await c.async_set_cover_tilt_position(tilt_position=target)
    c._async_set_compensated_tilt.assert_awaited_once_with(target)
    c._async_start_tilt.assert_not_called()


@pytest.mark.parametrize("tilt", [0, 100])
@pytest.mark.parametrize("target,expected", [(0, 0), (50, 100), (100, 100)])
async def test_pilot_granularity_respects_native_minimum_or_is_a_noop(
    hass: HomeAssistant,
    tilt: int,
    target: int,
    expected: int,
) -> None:
    c, gw = _native(
        hass, slat_offset=1, slat_time=0.5, slat_granularity=100, min_position_delta=2
    )
    c._attr_current_cover_position = 90
    c._attr_current_cover_tilt_position = tilt
    if tilt == expected:
        await c.async_set_cover_tilt_position(tilt_position=target)
    else:
        with pytest.raises(ServiceValidationError, match="min_position_delta=2"):
            await c.async_set_cover_tilt_position(tilt_position=target)
    gw.send.assert_not_called()
    gw.send_status_request.assert_not_called()
    assert c.current_cover_position == 90 and c.current_cover_tilt_position == tilt
    assert not c._is_tilting and c._compensation_cancel is None


@pytest.mark.parametrize(
    "tilt,target,expected,pre", [(0, 50, 100, 89), (0, 100, 100, 89), (100, 0, 0, 91)]
)
async def test_explicit_native_minimum_allows_pilot_tilt_without_height_drift(
    hass: HomeAssistant,
    tilt: int,
    target: int,
    expected: int,
    pre: int,
) -> None:
    c, gw = _native(
        hass, slat_offset=1, slat_time=0.5, slat_granularity=100, min_position_delta=2
    )
    c._attr_current_cover_position = 90
    c._attr_current_cover_tilt_position = tilt
    c._persist_calibration = MagicMock()
    await c.async_set_travel_time(min_position_delta=1)
    assert c._persist_calibration.call_args.args[0]["min_position_delta"] == 1
    gw.send.assert_not_called()
    _confirm_motion(c, gw, 90)
    assert await c.async_set_cover_tilt_position(tilt_position=target) is None
    result = c.extra_state_attributes["slat_compensation"]
    assert result["status"] == "done" and result["pre_position"] == pre
    assert result["tilt_position"] == expected and result["current_tilt_position"] == expected
    assert result["expected_position"] == result["current_position"] == 90
    assert result["position_error"] == 0 and c._tilt_duration == 0.5
    direction = 1 if expected == 100 else 2
    assert [str(call.args[0]) for call in gw.send.call_args_list] == [
        f"*#2*21*#11#001*{pre}##", f"*2*{direction}*21##", "*2*0*21##"
    ]


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

    async def confirmed(
        event: asyncio.Event | asyncio.Future[float], cancel: asyncio.Event, timeout: float,
    ) -> bool:
        c._attr_current_cover_tilt_position = tilt
        return True

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
    if action == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
    else:
        await asyncio.wait_for(task, 1)
    assert c.extra_state_attributes["slat_compensation"]["status"] == "interrupted"
    c._async_start_tilt.assert_not_called()


@pytest.mark.parametrize("phase", ["height", "tilt"])
@pytest.mark.parametrize("action", ["timeout", "cancel", "stop"])
async def test_abandoned_delivery_is_withdrawn(
    hass: HomeAssistant,
    phase: str,
    action: str,
    caplog: pytest.LogCaptureFixture,
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
        if action == "stop" or (action == "timeout" and phase == "tilt"):
            await asyncio.wait_for(task, 1)
        elif action == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 1)
        else:
            with pytest.raises(HomeAssistantError):
                await asyncio.wait_for(task, 1)
    assert written.cancelled() and not c._is_tilting
    assert c.extra_state_attributes["slat_compensation"]["status"] == "interrupted"
    if action == "timeout" and phase == "tilt":
        assert any(
            "Tilt pulse" in record.getMessage() and "was not delivered" in record.getMessage()
            for record in caplog.records
        )


@pytest.mark.parametrize("failure", ["cancel", "fail", "timeout"])
async def test_late_direction_failure_cannot_become_done_after_height_report(
    hass: HomeAssistant, failure: str,
) -> None:
    c, gw = _native(hass, slat_offset=0.5, slat_time=0.5)
    written = asyncio.get_running_loop().create_future()
    gw.send.return_value = written

    async def status(command: object) -> None:
        c.handle_event(OWNEvent.parse("*#2*21*10*10*50*001*0##"))

    gw.send_status_request.side_effect = status
    with patch("custom_components.myhome.cover.WRITE_TIMEOUT", 0.01):
        request = asyncio.create_task(c.async_set_cover_tilt_position(tilt_position=100))
        try:
            for _ in range(5):
                await asyncio.sleep(0)
            assert c._pulse_written is written and c._is_tilting
            if failure == "cancel":
                written.cancel()
            elif failure == "fail":
                written.set_exception(RuntimeError("late direction delivery failed"))
            await asyncio.wait_for(request, 1)
            c.handle_event(OWNEvent.parse("*#2*21*10*10*50*001*0##"))
            assert c.extra_state_attributes["slat_compensation"]["status"] == "interrupted"
            assert c.current_cover_tilt_position == 0
            gw.send_status_request.assert_not_called()
        finally:
            if not request.done():
                request.cancel()
            if not written.done():
                written.cancel()
            await asyncio.gather(request, return_exceptions=True)
            if (pulse_task := c._stop_task) is not None:
                c._cancel_stop_task()
                await asyncio.gather(pulse_task, return_exceptions=True)


@pytest.mark.parametrize("failure", ["cancel", "fail", "raise"])
@pytest.mark.parametrize("legacy_direction", [False, True])
async def test_failed_stop_receipt_keeps_settlement_but_never_reports_done(
    hass: HomeAssistant, failure: str, legacy_direction: bool, caplog: pytest.LogCaptureFixture,
) -> None:
    c, gw = _native(hass, slat_offset=0.5, slat_time=0.5)
    stopped = asyncio.get_running_loop().create_future()
    queued = asyncio.Event()

    async def send(command: object) -> asyncio.Future[float] | None:
        if str(command) == "*2*0*21##":
            queued.set()
            if failure == "raise":
                raise HomeAssistantError("stop send failed")
            return stopped
        if legacy_direction:
            return None
        c.handle_event(OWNEvent.parse(str(command)))
        written = asyncio.get_running_loop().create_future()
        written.set_result(time.monotonic())
        return written

    async def status(command: object) -> None:
        c.handle_event(OWNEvent.parse("*#2*21*10*10*50*001*0##"))

    gw.send.side_effect = send
    gw.send_status_request.side_effect = status
    request = asyncio.create_task(c.async_set_cover_tilt_position(tilt_position=100))
    try:
        await asyncio.wait_for(queued.wait(), 1)
        assert c._motor_started.is_set() is not legacy_direction
        if failure == "cancel":
            stopped.cancel()
        elif failure == "fail":
            stopped.set_exception(RuntimeError("stop delivery failed"))
        await asyncio.wait_for(request, 1)
        c.handle_event(OWNEvent.parse("*#2*21*10*10*50*001*0##"))
        assert c.extra_state_attributes["slat_compensation"]["status"] == "interrupted"
        assert c.current_cover_tilt_position == 100 and not c._is_tilting
        gw.send_status_request.assert_not_called()
        if failure == "raise":
            assert any(
                "Auto-stop tilt aborted" in record.getMessage() and "stop send failed" in record.getMessage()
                for record in caplog.records
            )
    finally:
        if not request.done():
            request.cancel()
        if not stopped.done():
            stopped.cancel()
        await asyncio.gather(request, return_exceptions=True)
        if (pulse_task := c._stop_task) is not None:
            c._cancel_stop_task()
            await asyncio.gather(pulse_task, return_exceptions=True)


async def test_cancelled_height_delivery_is_a_service_error(hass: HomeAssistant) -> None:
    c, gw = _native(hass, slat_offset=1)
    written = asyncio.get_running_loop().create_future()
    written.cancel()
    gw.send.return_value = written
    with pytest.raises(HomeAssistantError, match="cancelled before reaching the bus"):
        await c.async_set_cover_tilt_position(tilt_position=100)


async def test_failed_delivery_uses_existing_error_key(hass: HomeAssistant) -> None:
    c, _ = _native(hass)
    failure = RuntimeError("gateway disconnected")
    written = asyncio.get_running_loop().create_future()
    written.set_exception(failure)
    with pytest.raises(HomeAssistantError, match="gateway disconnected") as err:
        await c._wait_compensation_event(written, asyncio.Event(), 1)
    assert err.value.translation_key == "command_delivery_failed"
    assert err.value.__cause__ is failure


async def test_unconfirmed_stop_uses_existing_timeout_error_without_motion(hass: HomeAssistant) -> None:
    c, gw = _native(hass)
    with pytest.raises(HomeAssistantError) as err:
        await c._wait_compensation_event(asyncio.Event(), asyncio.Event(), 0)
    assert err.value.translation_key == "calibration_interrupted"
    assert err.value.translation_placeholders == {
        "name": c._display_name, "cause": "no stop status from the actuator within 0 s",
    }
    assert c.current_cover_position == 50 and c.current_cover_tilt_position == 0 and not c._is_tilting
    gw.send.assert_not_called()


async def test_failed_confirmation_wait_preserves_original_error(hass: HomeAssistant) -> None:
    c, _ = _native(hass)
    event = asyncio.Event()
    failure = RuntimeError("confirmation wait failed")
    with patch.object(event, "wait", AsyncMock(side_effect=failure)):
        with pytest.raises(RuntimeError, match="confirmation wait failed") as err:
            await c._wait_compensation_event(event, asyncio.Event(), 1)
    assert err.value is failure


async def test_pulse_wait_deadline_does_not_cancel_owned_task(hass: HomeAssistant) -> None:
    c, _ = _native(hass)
    pending = asyncio.create_task(asyncio.Event().wait())
    c._stop_task = pending
    try:
        with patch.multiple("custom_components.myhome.cover", WRITE_TIMEOUT=0, MAX_SLAT_TIME=0, ECHO_WINDOW=0):
            await c._async_wait_pulse_end()
        assert c._stop_task is pending and not pending.done()
    finally:
        c._cancel_stop_task()
        await asyncio.gather(pending, return_exceptions=True)


async def test_replacement_stop_task_receives_a_fresh_wait_budget(hass: HomeAssistant) -> None:
    c, _ = _native(hass)
    first_ready, final_ready = asyncio.Event(), asyncio.Event()
    first = asyncio.create_task(first_ready.wait())
    final = asyncio.create_task(final_ready.wait())
    c._stop_task = first
    clock = [0.0]
    waits: list[float] = []

    async def finish_task(
        tasks: set[asyncio.Task[bool]], *, timeout: float,
    ) -> tuple[set[asyncio.Task[bool]], set[asyncio.Task[bool]]]:
        waits.append(timeout)
        if tasks == {first}:
            c._stop_task = final
            clock[0] = 5
            first_ready.set()
        else:
            assert tasks == {final}
            final_ready.set()
        await asyncio.sleep(0)
        return tasks, set()

    try:
        with (
            patch.multiple("custom_components.myhome.cover", WRITE_TIMEOUT=1, MAX_SLAT_TIME=2, ECHO_WINDOW=3),
            patch("custom_components.myhome.cover.time.monotonic", side_effect=lambda: clock[0]),
            patch("custom_components.myhome.cover.asyncio.wait", side_effect=finish_task),
        ):
            await c._async_wait_pulse_end()
        assert waits == [6, 6] and final.done()
    finally:
        c._cancel_stop_task()
        for task in (first, final):
            if not task.done():
                task.cancel()
        await asyncio.gather(first, final, return_exceptions=True)


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
        await c.async_set_cover_tilt_position(tilt_position=50)
        assert c.extra_state_attributes["slat_compensation"]["status"] == "interrupted"
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
    ) -> bool:
        ready = await wait(event, cancel, timeout)
        if event is c._compensation_height_stopped and c._compensation_phase == "height":
            await c.async_stop_cover()
        return ready

    c._wait_compensation_event = AsyncMock(side_effect=stop_after_confirmation)
    c._async_start_tilt = AsyncMock()
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
    await asyncio.wait_for(task, 1)
    c._async_start_tilt.assert_not_called()
    assert [str(call.args[0]) for call in gw.send.call_args_list] == ["*2*0*21##"]


async def test_gateway_failure_is_visible_in_compensation_result(hass: HomeAssistant) -> None:
    c, gw = _native(hass, slat_offset=1)
    gw.send.side_effect = RuntimeError("gateway disconnected")
    with pytest.raises(RuntimeError, match="gateway disconnected"):
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
    await asyncio.wait_for(task, 1)
    assert c.extra_state_attributes["slat_compensation"]["status"] == "interrupted"
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


async def test_retarget_keeps_real_auto_stop_and_offset_feedback(hass: HomeAssistant) -> None:
    c, gw = _native(hass, slat_offset=0.5, slat_time=1)

    async def send(command: object) -> asyncio.Future[float]:
        if str(command) == "*2*1*21##":
            c.handle_event(OWNEvent.parse(str(command)))
        written = asyncio.get_running_loop().create_future()
        written.set_result(time.monotonic())
        return written

    async def status(command: object) -> None:
        c.handle_event(OWNEvent.parse("*#2*21*10*10*51*001*0##"))

    gw.send.side_effect = send
    gw.send_status_request.side_effect = status
    request = asyncio.create_task(c.async_set_cover_tilt_position(tilt_position=100))
    try:
        for _ in range(8):
            await asyncio.sleep(0)
        assert c._is_tilting and c._motor_started.is_set()
        cancel = c._compensation_cancel
        c._tilt_start_time = time.monotonic() - 0.2
        await c.async_set_cover_tilt_position(tilt_position=80)
        assert c._compensation_cancel is cancel and cancel is not None and not cancel.is_set()
        c.handle_event(OWNEvent.parse("*#2*21*10*10*51*001*0##"))
        assert c._is_tilting and c._stop_task is not None
        await asyncio.wait_for(request, 2)
        result = c.extra_state_attributes["slat_compensation"]
        assert result["status"] == "done"
        assert result["current_tilt_position"] == pytest.approx(80, abs=1)
        assert result["tilt_position"] == 80 and result["current_position"] == 51
        assert result["expected_position"] == pytest.approx(50.4)
        assert [str(call.args[0]) for call in gw.send.call_args_list] == [
            "*2*1*21##", "*2*0*21##",
        ]
        gw.send_status_request.assert_awaited_once()
    finally:
        if not request.done():
            request.cancel()
        await asyncio.gather(request, return_exceptions=True)
        if (pulse_task := c._stop_task) is not None:
            c._cancel_stop_task()
            await asyncio.gather(pulse_task, return_exceptions=True)


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


@pytest.mark.parametrize("action", ["set", "reset"])
async def test_slat_calibration_during_pulse_does_not_interrupt_motion(
    hass: HomeAssistant, action: str,
) -> None:
    c, gw = _native(hass, slat_offset=0.5, slat_time=2)
    c._persist_calibration = MagicMock()
    c._device_config = MagicMock(return_value={
        "slat_time": 2, "slat_offset": 0.5, "min_position_delta": 1,
    })
    _confirm_motion(c, gw, 50)
    started, release = asyncio.Event(), asyncio.Event()

    async def finish_after_calibration() -> None:
        started.set()
        await release.wait()
        await _finish_tilt(c)

    c._async_wait_pulse_end = AsyncMock(side_effect=finish_after_calibration)
    request = asyncio.create_task(c.async_set_cover_tilt_position(tilt_position=100))
    try:
        await asyncio.wait_for(started.wait(), 1)
        assert c._is_tilting and c._motor_started.is_set()
        if action == "set":
            await c.async_set_travel_time(slat_time=2)
        else:
            await c.async_reset_travel_time()
        release.set()
        await asyncio.wait_for(request, 1)
        assert c.extra_state_attributes["slat_compensation"]["status"] == "done"
        assert [str(call.args[0]) for call in gw.send.call_args_list] == [
            "*2*1*21##", "*2*0*21##",
        ]
    finally:
        release.set()
        if not request.done():
            request.cancel()
        await asyncio.gather(request, return_exceptions=True)
        if (pulse_task := c._stop_task) is not None:
            c._cancel_stop_task()
            await asyncio.gather(pulse_task, return_exceptions=True)


async def test_feasible_in_flight_tilt_retargets_existing_pulse(hass: HomeAssistant) -> None:
    c, gw = _native(hass, slat_offset=0.5, slat_time=1)
    _confirm_motion(c, gw, 50)
    started, release = asyncio.Event(), asyncio.Event()

    async def finish_retargeted_pulse() -> None:
        started.set()
        await release.wait()
        await _finish_tilt(c)

    c._async_wait_pulse_end = AsyncMock(side_effect=finish_retargeted_pulse)
    request = asyncio.create_task(c.async_set_cover_tilt_position(tilt_position=100))
    try:
        await asyncio.wait_for(started.wait(), 1)
        assert c._tilt_start_time is not None
        with patch("custom_components.myhome.cover.time.monotonic", return_value=c._tilt_start_time + 0.2):
            await c.async_set_cover_tilt_position(tilt_position=80)
        assert c._is_tilting and c._tilt_target_position == 80
        assert c._tilt_duration == 0.8
        release.set()
        await asyncio.wait_for(request, 1)
        assert c.current_cover_tilt_position == 80
        assert [str(call.args[0]) for call in gw.send.call_args_list] == [
            "*2*1*21##", "*2*0*21##",
        ]
    finally:
        release.set()
        if not request.done():
            request.cancel()
        await asyncio.gather(request, return_exceptions=True)
        if (pulse_task := c._stop_task) is not None:
            c._cancel_stop_task()
            await asyncio.gather(pulse_task, return_exceptions=True)


async def test_stop_supersedes_active_retarget_waiting_for_lock(hass: HomeAssistant) -> None:
    c, gw = _native(hass, slat_offset=0.5, slat_time=2)
    _confirm_motion(c, gw, 50)
    started, release = asyncio.Event(), asyncio.Event()

    async def wait_for_manual_stop() -> None:
        started.set()
        await release.wait()

    c._async_wait_pulse_end = AsyncMock(side_effect=wait_for_manual_stop)
    original = asyncio.create_task(c.async_set_cover_tilt_position(tilt_position=100))
    queued: asyncio.Task[None] | None = None
    try:
        await asyncio.wait_for(started.wait(), 1)
        async with c._tilt_lock:
            sequence = c._tilt_request_seq
            queued = asyncio.create_task(c.async_set_cover_tilt_position(tilt_position=80))
            await asyncio.sleep(0)
            assert not queued.done() and c._tilt_request_seq == sequence + 1
            await c.async_stop_cover()
        release.set()
        await asyncio.wait_for(asyncio.gather(original, queued), 1)
        assert c.extra_state_attributes["slat_compensation"]["status"] == "interrupted"
        assert c._tilt_target_position == 100 and not c._is_tilting and c._stop_task is None
        assert [str(call.args[0]) for call in gw.send.call_args_list] == [
            "*2*1*21##", "*2*0*21##",
        ]
    finally:
        release.set()
        for request in (original, queued):
            if request is not None:
                if not request.done():
                    request.cancel()
                await asyncio.gather(request, return_exceptions=True)
        if (pulse_task := c._stop_task) is not None:
            c._cancel_stop_task()
            await asyncio.gather(pulse_task, return_exceptions=True)


async def test_retarget_requiring_height_move_preserves_original_pulse(hass: HomeAssistant) -> None:
    c, gw = _native(hass, slat_offset=10, slat_time=2)
    _confirm_motion(c, gw, 50)
    started, release = asyncio.Event(), asyncio.Event()

    async def finish_original_pulse() -> None:
        started.set()
        await release.wait()
        await _finish_tilt(c)

    c._async_wait_pulse_end = AsyncMock(side_effect=finish_original_pulse)
    request = asyncio.create_task(c.async_set_cover_tilt_position(tilt_position=100))
    try:
        await asyncio.wait_for(started.wait(), 1)
        cancel, plan, sequence = c._compensation_cancel, c._compensation_plan, c._tilt_request_seq
        assert plan is not None and plan.pre_position == 40 and plan.start_tilt == 0
        assert plan.expected_position - plan.position_error == 50
        assert c._tilt_start_time is not None
        with patch("custom_components.myhome.cover.time.monotonic", return_value=c._tilt_start_time + 1.4):
            assert c._interpolated_tilt(time.monotonic()) == 70
            with pytest.raises(ServiceValidationError) as err:
                await c.async_set_cover_tilt_position(tilt_position=71)
        assert err.value.translation_key == "cover_busy_moving"
        assert c._compensation_cancel is cancel and cancel is not None and not cancel.is_set()
        assert c._compensation_plan is plan and c._tilt_request_seq == sequence
        assert c._is_tilting and c._tilt_target_position == 100
        release.set()
        await asyncio.wait_for(request, 1)
        assert c.extra_state_attributes["slat_compensation"]["status"] == "done"
    finally:
        release.set()
        if not request.done():
            request.cancel()
        await asyncio.gather(request, return_exceptions=True)
        if (pulse_task := c._stop_task) is not None:
            c._cancel_stop_task()
            await asyncio.gather(pulse_task, return_exceptions=True)


async def test_own_native_target_echo_preserves_compensation(hass: HomeAssistant) -> None:
    c, gw = _native(hass, slat_offset=1)
    _confirm_motion(c, gw, 50)
    send = gw.send.side_effect

    async def send_with_target_echo(command: object) -> asyncio.Future[float]:
        if str(command) == "*#2*21*#11#001*49##":
            c.handle_event(OWNEvent.parse("*#2*21*#11#001#1*49##"))
            assert c._compensation_cancel is not None and not c._compensation_cancel.is_set()
        return await send(command)

    gw.send.side_effect = send_with_target_echo
    await c.async_set_cover_tilt_position(tilt_position=100)
    assert c.extra_state_attributes["slat_compensation"]["status"] == "done"


async def test_delivered_command_after_takeover_cannot_resurrect_pulse(hass: HomeAssistant) -> None:
    c, gw = _native(hass, slat_offset=1)
    enqueued, release = asyncio.Event(), asyncio.Event()
    written = asyncio.get_running_loop().create_future()
    written.set_result(time.monotonic())

    async def send(command: object) -> asyncio.Future[float]:
        if str(command) == "*2*1*21##":
            enqueued.set()
            await release.wait()
            return written
        stopped = asyncio.get_running_loop().create_future()
        stopped.set_result(time.monotonic())
        return stopped

    gw.send.side_effect = send
    request = asyncio.create_task(c.async_set_cover_tilt_position(tilt_position=50))
    try:
        await asyncio.wait_for(enqueued.wait(), 1)
        await c.async_stop_cover()
        release.set()
        await asyncio.wait_for(request, 1)
        assert not written.cancelled()
        assert not c._is_tilting and c._stop_task is None
    finally:
        release.set()
        if not request.done():
            request.cancel()
        await asyncio.gather(request, return_exceptions=True)


async def test_stale_request_failure_does_not_clear_newer_compensation(hass: HomeAssistant) -> None:
    c, gw = _native(hass, slat_offset=1)
    _confirm_motion(c, gw, 50)
    send = gw.send.side_effect
    enqueued, fail_old = asyncio.Event(), asyncio.Event()
    started, release = asyncio.Event(), asyncio.Event()

    async def fail_stale_height(command: object) -> asyncio.Future[float]:
        if str(command) == "*#2*21*#11#001*49##":
            enqueued.set()
            await fail_old.wait()
            raise RuntimeError("stale gateway failure")
        return await send(command)

    async def finish_newer_pulse() -> None:
        started.set()
        await release.wait()
        await _finish_tilt(c)

    gw.send.side_effect = fail_stale_height
    c._async_wait_pulse_end = AsyncMock(side_effect=finish_newer_pulse)
    old = asyncio.create_task(c.async_set_cover_tilt_position(tilt_position=100))
    newer: asyncio.Task[None] | None = None
    try:
        await asyncio.wait_for(enqueued.wait(), 1)
        newer = asyncio.create_task(c.async_set_cover_tilt_position(tilt_position=50))
        await asyncio.wait_for(started.wait(), 1)
        cancel, plan = c._compensation_cancel, c._compensation_plan
        fail_old.set()
        with pytest.raises(RuntimeError, match="stale gateway failure"):
            await asyncio.wait_for(old, 1)
        assert c._compensation_cancel is cancel and cancel is not None and not cancel.is_set()
        assert c._compensation_plan is plan and c._compensation_phase == "tilt"
        release.set()
        await asyncio.wait_for(newer, 1)
        assert c.current_cover_tilt_position == 50
    finally:
        fail_old.set()
        release.set()
        for request in (old, newer):
            if request is not None:
                if not request.done():
                    request.cancel()
                await asyncio.gather(request, return_exceptions=True)
        if (pulse_task := c._stop_task) is not None:
            c._cancel_stop_task()
            await asyncio.gather(pulse_task, return_exceptions=True)


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
