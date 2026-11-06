"""Pure slat target validation and single-run height compensation planning."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .const import (
    CONF_SLAT_GRANULARITY,
    CONF_SLAT_OFFSET,
    CONF_SLAT_OFFSET_DOWN,
    CONF_SLAT_OFFSET_UP,
    CONF_SLAT_TIME,
    CONF_SLAT_TIME_DOWN,
    CONF_SLAT_TIME_UP,
    MAX_SLAT_GRANULARITY,
    MAX_SLAT_OFFSET,
    MAX_SLAT_TIME,
    MIN_SLAT_GRANULARITY,
    MIN_SLAT_OFFSET,
    MIN_SLAT_TIME,
)

SLAT_SETTINGS = (
    CONF_SLAT_TIME,
    CONF_SLAT_TIME_UP,
    CONF_SLAT_TIME_DOWN,
    CONF_SLAT_OFFSET,
    CONF_SLAT_OFFSET_UP,
    CONF_SLAT_OFFSET_DOWN,
    CONF_SLAT_GRANULARITY,
)


def validate_slat_setting(name: str, value: object) -> float | int:
    """Validate a finite calibration value without truncating fractional steps.

    :param name: Slat setting key
    :param value: Requested calibration value
    :returns: Validated numeric value
    :raises ValueError: If the setting is invalid
    """
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    low, high = (
        (MIN_SLAT_TIME, MAX_SLAT_TIME)
        if name.startswith(CONF_SLAT_TIME)
        else (MIN_SLAT_OFFSET, MAX_SLAT_OFFSET)
    )
    if name == CONF_SLAT_GRANULARITY:
        low, high = MIN_SLAT_GRANULARITY, MAX_SLAT_GRANULARITY
    if not math.isfinite(number) or not low <= number <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    if name == CONF_SLAT_GRANULARITY:
        if not number.is_integer():
            raise ValueError("slat_granularity must be a whole percentage")
        return int(number)
    return number


def snap_tilt_target(target: float, granularity: int) -> int:
    """Round a requested target to the nearest step, ties up; include both ends.

    :param target: Requested tilt percentage
    :param granularity: Whole-percentage target step
    :returns: Rounded target, not a rounded current estimate
    """
    target = max(0.0, min(100.0, target))
    lower = int(target // granularity) * granularity
    upper = min(100, lower + granularity)
    return upper if target - lower >= upper - target else lower


@dataclass(frozen=True)
class SlatPlan:
    """A native pre-position followed by at most one slat pulse."""

    pre_position: int
    start_tilt: int
    expected_position: float
    position_error: float
    limited: bool
    move: bool


def slat_height_change(start: int, target: int, offset_up: float, offset_down: float) -> float:
    """Calculate signed height change proportional to the remaining slat travel.

    :param start: Current or predicted slat angle
    :param target: Requested slat angle
    :param offset_up: Full opening height change
    :param offset_down: Full closing height change, negated when closing
    :returns: Expected height displacement
    """
    return (target - start) / 100.0 * (offset_up if target >= start else offset_down)


def plan_tilt_compensation(
    current: int,
    current_tilt: int | None,
    target_tilt: int,
    *,
    offset_up: float,
    offset_down: float,
    time_up: float,
    time_down: float,
    min_delta: int,
    min_pulse: float,
) -> SlatPlan:
    """Choose a no-run/up/down candidate that best preserves reported height.

    :param current: Authoritative bus height
    :param current_tilt: Current unrounded slat estimate
    :param target_tilt: Already rounded requested tilt
    :param offset_up: Full opening offset
    :param offset_down: Full closing offset
    :param time_up: Full opening time
    :param time_down: Full closing time
    :param min_delta: Minimum native height movement (never widened)
    :param min_pulse: Minimum executable motor pulse
    :returns: Best valid single-run plan
    :raises ValueError: If no meaningful single-run plan is executable
    """
    candidates: list[SlatPlan] = []
    short_pulse = False
    for direction, start in ((0, current_tilt), (1, 100), (-1, 0)):
        if start is None:
            continue
        change = slat_height_change(start, target_tilt, offset_up, offset_down)
        raw_pre = current - change
        pre = current if direction == 0 else int(math.floor(max(0.0, min(100.0, raw_pre)) + 0.5))
        if direction and ((pre - current) * direction <= 0 or abs(pre - current) < min_delta):
            continue
        final = max(0.0, min(100.0, pre + change))
        error = final - current
        # Tilt alone is useful only if it reaches the requested height within native rounding.
        if direction == 0 and abs(error) > 0.5:
            continue
        duration = (
            abs(target_tilt - start) / 100.0 * (time_up if target_tilt > start else time_down)
        )
        if 0 < duration < min_pulse:
            short_pulse = True
            continue
        candidates.append(
            SlatPlan(pre, start, final, error, not 0 <= raw_pre <= 100, bool(direction))
        )
    if not candidates:
        reason = (
            "remaining tilt pulse is below min_motor_pulse"
            if short_pulse
            else "minimum height movement or direction"
        )
        raise ValueError(f"No executable single height run plus tilt: {reason}; no movement sent")
    return min(
        candidates, key=lambda plan: (abs(plan.position_error), abs(plan.pre_position - current))
    )
