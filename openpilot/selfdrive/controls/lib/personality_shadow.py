import numpy as np
from openpilot.cereal import log
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.accel_boost import AccelBoost

PERSONALITIES = [log.LongitudinalPersonality.aggressive, log.LongitudinalPersonality.standard, log.LongitudinalPersonality.relaxed]

# The model's action falls as the car goes faster than it wants, like a weak spring: about 0.01 m/s^2 less for every m/s,
# so it takes the model over a minute to pull the speed back. Its plan, averaged over 10 s, moves by about 5 s of that.
SHADOW_K = 0.01  # 1/s
SHADOW_PLAN_SHIFT = 5.0 * SHADOW_K
# While the plan heads for a speed away from the car's (pulling away, slowing for something), the model steers toward that
# speed: a personality further from it gets a bigger push, the model's accel scaled by how much further it has to go. It
# blends in from this far between the car and the plan's speed in 10 s, fully at twice that
SHADOW_TARGET_GAP = 2.0  # m/s
SHADOW_TARGET_RATIO_MAX = 3.0

# Switching to a less aggressive personality, or engaging, slows toward that personality's shadow speed over 5 s
SWITCH_TIME = 5.0  # s
SWITCH_RAMP = 1.0  # s to ease the slowdown in and out, 0.5 m/s^3 at most
SWITCH_ACCEL_MAX = 0.5  # m/s^2
SWITCH_SPEED_FRAC = 0.1  # at most 10% of the speed over the 5 s

# The model braking at least this hard is slowing for something every personality slows for (a turn, a stop, a curve, a slower
# car): every personality moves with the car, and the car's own braking closes the gap first. A slowdown the driver skips
# (a turn not taken) leaves the gaps alone. Lighter pushback is the spring above
BRAKE_DRAIN_ACCEL = 0.3  # m/s^2
MIN_SPEED = 2.0  # m/s, shadows follow the car below this, and in reverse
LEAD_CAP_TIME = 3.0  # s, a lead this close, or one openpilot is following, caps every personality at its speed
LEAD_CAP_DECEL = 1.5  # m/s^2, a personality above the lead slows to it at this rate, so a noisy lead reading barely moves it


def _raw(personality):
  return personality.raw if hasattr(personality, 'raw') else int(personality)


class PersonalityShadow:
  """Tracks the speed each personality would be going had it been driving the whole time, from the model's action and plan
  with a small correction for the speed gap to the car. Shadows run whether or not openpilot is engaged."""

  def __init__(self, dt=DT_MDL):
    self.dt = dt
    self.speeds = {_raw(p): None for p in PERSONALITIES}
    self.boosts = {_raw(p): AccelBoost(dt) for p in PERSONALITIES}
    self.personality = None
    self.enabled = False
    self.switch_t = 0.0
    self.switch_level = 0.0  # m/s^2 the slowdown holds at, 0 when not slowing
    self.switch_target = 0.0  # m/s it slows toward
    self.switch_accel = 0.0  # m/s^2 it took off the last output

  def _start_switch(self, personality, v_ego):
    target = self.speeds[personality]
    gap = v_ego - target if target is not None else 0.0
    if gap <= 0.0:
      self.switch_level = 0.0
      return
    self.switch_t = 0.0
    self.switch_target = target
    self.switch_level = min(SWITCH_ACCEL_MAX, gap / SWITCH_TIME, SWITCH_SPEED_FRAC * v_ego / SWITCH_TIME)

  def update(self, personality, v_ego, a_ego, a_e2e, plan_ahead, curve_offset, engaged, enabled, gas_pressed, standstill,
             reverse=False, v_lead=None, v_cruise=None, v_plan_end=None):
    """a_e2e: the model's accel with the curve offset, before any boost. v_plan_end: the plan's speed in 10 s.
    engaged: openpilot is controlling the speed. enabled:
    openpilot is on, also while the driver overrides with the gas. v_lead: the speed of a lead close enough to follow.
    v_cruise: the set speed while engaged, no personality goes faster. It never pulls one below the car: a set speed below
    the car is slowed for by the cruise control anyway, and right after engaging it can still be the old one."""
    p = _raw(personality)
    override = enabled and gas_pressed

    # slow toward a less aggressive personality on switching to it, or on engaging, before its shadow joins the car
    if enabled and not override and (not self.enabled or (self.personality is not None and p > self.personality)):
      self._start_switch(p, v_ego)
    elif self.personality is not None and p < self.personality:
      self.switch_level = 0.0  # a more aggressive personality cancels it
    self.personality, self.enabled = p, enabled

    if self.switch_level > 0.0:
      self.switch_t += self.dt
      if not engaged or self.switch_t >= SWITCH_TIME or v_ego <= self.switch_target:
        self.switch_level = 0.0

    if v_ego < MIN_SPEED or standstill or reverse:
      for q in self.speeds:
        self.speeds[q] = v_ego
        self.boosts[q].reset()
      return

    # the personality that's driving is the car, also while the driver overrides with the gas
    driving = engaged or override
    boost_driving = self.boosts[p].boost(p, a_e2e, plan_ahead, curve_offset) if driving else 0.0
    # the driver pressing the gas wants to go faster: every personality goes faster with them, nothing builds up to take back
    extra = max(a_ego - (a_e2e + boost_driving), 0.0) if override else 0.0

    for q, v_s in self.speeds.items():
      if v_s is None or (driving and q == p):
        self.speeds[q] = v_ego
        continue
      gap = v_ego - v_s
      a_s = self._shadow_accel(a_e2e, v_ego, v_s, v_plan_end)
      boost = self.boosts[q].boost(q, a_s, plan_ahead + SHADOW_PLAN_SHIFT * gap, curve_offset)
      if a_e2e < -BRAKE_DRAIN_ACCEL:
        v_s += a_ego * self.dt + float(np.sign(gap)) * min(abs(gap), max(-a_ego, 0.0) * self.dt)
      else:
        v_s += (a_s + boost + extra) * self.dt
      v_s = max(v_s, 0.0)
      if v_lead is not None:
        v_s = max(min(v_s, max(v_lead, 0.0)), v_s - LEAD_CAP_DECEL * self.dt)
      if v_cruise is not None:
        v_s = min(v_s, max(v_cruise, v_ego))
      self.speeds[q] = v_s

  @staticmethod
  def _shadow_accel(a_e2e, v_ego, v_s, v_plan_end):
    """The model's accel for a personality at v_s: the weak spring, or toward the plan's speed while it heads somewhere."""
    spring = a_e2e + SHADOW_K * (v_ego - v_s)
    if v_plan_end is None:
      return spring
    to_target = v_plan_end - v_ego
    w = float(np.clip((abs(to_target) - SHADOW_TARGET_GAP) / SHADOW_TARGET_GAP, 0.0, 1.0))
    if w == 0.0 or a_e2e * to_target <= 0.0:
      return spring
    toward = a_e2e * float(np.clip((v_plan_end - v_s) / to_target, 0.0, SHADOW_TARGET_RATIO_MAX))
    return w * toward + (1.0 - w) * spring

  def apply(self, output_a_target_e2e):
    """Slow toward the switched-to personality: decel to at least the slowdown's level, taking off at most that much."""
    if self.switch_level <= 0.0:
      self.switch_accel = 0.0
      return output_a_target_e2e
    ramp = float(np.clip(min(self.switch_t, SWITCH_TIME - self.switch_t) / SWITCH_RAMP, 0.0, 1.0))
    level = self.switch_level * ramp
    out = min(output_a_target_e2e, max(output_a_target_e2e - level, -level))
    self.switch_accel = out - output_a_target_e2e
    return out

  def shadow_speeds(self):
    # by personality, in the enum's order: aggressive, standard, relaxed
    return [float(self.speeds[_raw(p)] or 0.0) for p in PERSONALITIES]
