import numpy as np
from opendbc.car.structs import car
from openpilot.cereal import log
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.accel_boost import AccelBoost, get_plan_ahead

STANDARD = int(log.LongitudinalPersonality.standard)  # aggressive=0, standard=1, relaxed=2

SPRING_K, PLAN_GAP, BRAKE_DRAIN = 0.01, 2.0, 0.3             # 1/s spring, m/s plan blend, m/s^2 hard brake
MIN_SPEED, LEAD_TIME = 2.0, 2.0                              # m/s reset floor, s lead time-gap cap
SWITCH_TIME, SWITCH_RAMP, SWITCH_PRESS_RAMP = 5.0, 1.0, 0.5  # s detach window, end fade, slowdown slew
SWITCH_ACCEL_MAX, SWITCH_MARGIN = 0.5, 0.5                   # m/s^2 max decel, m/s margin
SWITCH_SPEED_FRAC, SWITCH_SPEED_CAP = 0.1, 2.5               # per-level fraction of v0, m/s run cap


def _shadow_accel(a_e2e: float, v_ego: float, v_s: float, v_plan_end: float | None) -> float:
  spring = a_e2e + SPRING_K * (v_ego - v_s)
  if v_plan_end is None or a_e2e * (v_plan_end - v_ego) <= 0.0:
    return spring
  to_plan = v_plan_end - v_ego
  w = float(np.clip((abs(to_plan) - PLAN_GAP) / PLAN_GAP, 0.0, 1.0))
  return spring + w * (a_e2e * float(np.clip((v_plan_end - v_s) / to_plan, 0.0, 3.0)) - spring)


class PersonalityShadow:
  """Tracks each personality's counterfactual speed and drives a smooth, bounded switch response."""

  def __init__(self, dt: float = DT_MDL):
    self.dt = dt
    self.speeds = [0.0] * 3
    self.boosts = [AccelBoost(dt) for _ in range(3)]
    self.personality, self._initialized, self.driven = STANDARD, False, False
    self.remaining, self.from_p, self.v0 = 0.0, STANDARD, 0.0  # detached window & run origin
    self.direction, self.strength, self.taking, self.switch_accel = 0, 0.0, 0.0, 0.0

  def _target(self, p: int, direction: int) -> float:
    levels = p - self.from_p if direction < 0 else self.from_p - p
    cap = min(SWITCH_SPEED_CAP, SWITCH_SPEED_FRAC * max(levels, 0) * self.v0)
    return max(self.speeds[p], self.v0 - cap) if direction < 0 else min(self.speeds[p], self.v0 + cap)

  def update(self, sm, a_e2e: float, curve_offset: float, engaged: bool, v_cruise: float | None = None) -> None:
    cs, model, lead = sm['carState'], sm['modelV2'], sm['radarState'].leadOne
    self.step(sm['selfdriveState'].personality, cs.vEgo, a_e2e, get_plan_ahead(model.position, cs.vEgo), cs.aEgo,
              curve_offset, engaged, sm['selfdriveState'].enabled and cs.gasPressed,
              cs.standstill or cs.gearShifter == car.CarState.GearShifter.reverse,
              (lead.dRel, lead.vLeadK) if lead.present else None, v_cruise,
              model.velocity.x[-1] if len(model.velocity.x) else None)

  def step(self, personality, v_ego: float, a_e2e: float, plan_ahead: float, a_ego: float = 0.0,
           curve_offset: float = 0.0, engaged: bool = True, override: bool = False, reset: bool = False,
           lead: tuple[float, float] | None = None, v_cruise: float | None = None, v_plan_end: float | None = None) -> None:
    p = int(getattr(personality, 'raw', personality))
    if not self._initialized:
      self._initialized, self.personality, self.speeds = True, p, [float(v_ego)] * 3

    # 1. Switch response: detach on press, slew slowdown and ease out toward target / window end
    prev_p, self.personality = self.personality, p
    if not engaged or override:
      self.remaining, self.direction, self.strength, self.taking, self.switch_accel = 0.0, 0, 0.0, 0.0, 0.0
    else:
      if p != prev_p:
        if self.remaining == 0.0:
          self.from_p, self.v0 = prev_p, v_ego
        self.remaining = SWITCH_TIME
        d = -1 if self.speeds[p] < v_ego else 1
        self.direction = d if d * (self._target(p, d) - v_ego) > SWITCH_MARGIN else 0
      self.remaining = 0.0 if self.remaining <= self.dt + 1e-6 else self.remaining - self.dt
      to_go = self.direction * (self._target(p, self.direction) - v_ego) if self.direction else 0.0
      if self.remaining == 0.0 or to_go <= 0.0:
        self.direction = 0
      self.strength = min(to_go / SWITCH_MARGIN, self.remaining / SWITCH_RAMP, 1.0) if self.direction else 0.0
      want = SWITCH_ACCEL_MAX * self.strength if self.direction < 0 else 0.0
      slew = SWITCH_ACCEL_MAX * self.dt / SWITCH_PRESS_RAMP
      self.taking = float(np.clip(want, self.taking - slew, self.taking + slew))

    # 2. Shadow speeds: integrate each personality's counterfactual speed with physical limits
    self.driven = engaged or override
    if v_ego < MIN_SPEED or reset:
      self.speeds = [float(v_ego)] * 3
      for b in self.boosts:
        b.reset()
      return

    car_p = p if self.driven and self.remaining == 0.0 else None
    if car_p is not None:
      self.speeds[car_p] = float(v_ego)
    accels = []
    for q, v_s in enumerate(self.speeds):
      a_s = _shadow_accel(a_e2e, v_ego, v_s, v_plan_end)
      accels.append(a_s + self.boosts[q].boost(q, a_s, plan_ahead + 5.0 * SPRING_K * (v_ego - v_s), curve_offset))
    extra = max(a_ego - accels[p], 0.0) if override else 0.0

    for q, v_s in enumerate(self.speeds):
      if q == car_p:
        continue
      if a_e2e < -BRAKE_DRAIN:
        closing = max(-a_ego, 0.0) * self.dt
        v_s += a_ego * self.dt + float(np.clip(v_ego - v_s, -closing, closing))
      else:
        v_s += (accels[q] + extra) * self.dt
      if lead is not None and lead[0] < LEAD_TIME * max(v_s, 1.0):
        v_s = min(v_s, max(v_ego, lead[1]))
      if engaged and v_cruise is not None:
        v_s = min(v_s, max(v_cruise, v_ego))
      if car_p is not None and q > car_p:
        v_s = min(v_s, v_ego)
      self.speeds[q] = max(v_s, 0.0)

  @property
  def bias_scale(self) -> float:
    return 1.0 + (self.strength if self.direction > 0 and self.taking == 0.0 else 0.0)

  def apply(self, a_e2e: float) -> float:
    out = min(a_e2e, max(a_e2e - self.taking, -self.taking))
    self.switch_accel = out - a_e2e
    return out

  def shadow_speeds(self) -> list[float]:
    v = [float(x) for x in self.speeds]
    if self.direction < 0:
      v[self.personality] = self._target(self.personality, -1)
    anchor = self.personality if self.driven else STANDARD
    for i in range(anchor - 1, -1, -1):
      v[i] = max(v[i], v[i + 1])
    for i in range(anchor + 1, len(v)):
      v[i] = min(v[i], v[i - 1])
    return v
