"""
Speed / accel caps derived from hazard distance for RoadPass → longitudinal planner.

Uses the same warning radius as HazardFetcher buckets; closer hazards request lower
set-speed ceilings and gentler positive acceleration (conservative comfort band).
"""
from openpilot.common.constants import CV
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX, V_CRUISE_UNSET

# Minimum speed cap (m/s) so we never command a near-standstill on highway-like roads.
_MIN_V_CAP_MS = 4.0


def cruise_kph_for_limits(car_state) -> float:
  """Aligned with LongitudinalPlanner: respect vCruise or fall back to a sensible floor."""
  v_kph = car_state.vCruise
  if v_kph == V_CRUISE_UNSET:
    v_kph = max(car_state.vEgo * CV.MS_TO_KPH, 15.0)
  return float(min(v_kph, V_CRUISE_MAX))


def longitudinal_limits_from_hazard(
    v_cruise_kph: float,
    distance_m: float,
    warn_distance_m: int,
) -> tuple[float, float]:
  """
  Return (maxSpeedMs, maxPositiveAccelMs2) for uiDebug → planner.

  `t` is 0 at the hazard, 1 at the outer edge of the warning zone.
  """
  v_cruise_ms = v_cruise_kph * CV.KPH_TO_MS
  if warn_distance_m <= 0:
    return v_cruise_ms, 1.2

  t = max(0.0, min(1.0, distance_m / float(warn_distance_m)))
  # Goal: cap toward ~40% of set speed near the hazard, full set speed at zone edge.
  v_cap = max(_MIN_V_CAP_MS, v_cruise_ms * (0.40 + 0.60 * t))
  # Conservative: softer accel when the hazard is closer.
  a_cap = 0.35 + 0.85 * t
  return v_cap, a_cap
