import numpy as np
from dataclasses import dataclass

from openpilot.cereal import log

Personality = log.LongitudinalPersonality

# desires that mean a turn or a lane split is coming
TURN_KEEP_DESIRES = (1, 2, 5, 6)  # turnLeft, turnRight, keepLeft, keepRight
DESIRE_WIDTH = 8

PLAN_SLOPE_T = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
PLAN_ACCEL_T = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0])
CURVE_T = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])


@dataclass
class PersonalityParams:
  a_lift_max: float  # m/s^2, most accel the lift can ask for
  jerk_up: float  # m/s^3, how fast the lift ramps in
  headroom_ratio: float  # how far above the model chosen speed the lift may go
  headroom_abs: float  # m/s, added to the ratio headroom
  a_lat_max: float  # m/s^2, curve speed cap


PERSONALITY_PARAMS = {
  int(Personality.aggressive): PersonalityParams(0.8, 0.8, 0.35, 2.0, 2.5),
  int(Personality.standard): PersonalityParams(0.6, 0.6, 0.25, 1.5, 2.0),
  int(Personality.relaxed): PersonalityParams(0.4, 0.4, 0.15, 1.0, 1.6),
}


@dataclass
class SpeedIntentConfig:
  # plan shape: worst average accel from now to any plan point in 1-6 s
  plan_slope_bp: tuple[float, float] = (-0.45, -0.2)
  plan_slope_bp_user: tuple[float, float] = (-0.6, -0.3)  # when the user asked for more speed
  plan_accel_veto: float = -0.8  # any planned accel below this in 0-4 s is a hard veto
  e2e_accel_veto: float = -0.4  # the model is braking now
  allow_flip: bool = False  # when False, a model slowdown can at most be held at 0, never turned into accel
  # leads
  lead_gap_bp: tuple[float, float] = (2.0, 4.0)  # s, time gap
  lead_gap_closing_bp: tuple[float, float] = (3.0, 6.0)
  lead_gap_veto: float = 1.5
  model_lead_prob: float = 0.5
  model_lead_gap: float = 4.0
  # model meta outputs
  desire_bp: tuple[float, float] = (0.15, 0.4)
  brake_prob_bp: tuple[float, float] = (0.15, 0.35)
  blinker_speed_bp: tuple[float, float] = (11.0, 15.0)  # m/s, blinker below this is a turn
  speed_bp: tuple[float, float] = (12.0, 16.0)  # m/s, fleet data: lifting below ~15 m/s gets as many brakes as overrides
  # tracking
  k_v: float = 0.4  # 1/s, accel per m/s of speed error
  jerk_down: float = 2.0  # m/s^3
  curve_decel: float = 0.7  # m/s^2, comfortable decel to reach a curve speed
  v_user_decay: float = 0.02  # m/s per s
  v_user_clear_drop: float = 5.0  # m/s, forget the user speed after a slowdown this big
  set_speed_gain: float = 1.0  # m/s of intent per m/s of set speed increase
  lead_clear_gap: float = 4.0  # s


class SpeedIntent:
  def __init__(self, config: SpeedIntentConfig | None = None):
    self.c = config or SpeedIntentConfig()
    self.lift = 0.0
    self.output_lift = 0.0
    self.anchor = 0.0
    self.v_user = 0.0
    self.v_pre_lead = 0.0
    self.prev_enabled = False
    self.prev_gas = False
    self.prev_v_cruise = 0.0
    self.prev_lead_limited = False
    self.debug: dict[str, float] = {}

  def _freedom(self, v_ego, plan_t, plan_v, accel_t, plan_a, desire_state, desire_pred, brake_prob, blinker,
               lead_present, lead_drel, lead_vlead, model_lead_prob, model_lead_x, user_active):
    c = self.c
    dv = np.interp(PLAN_SLOPE_T, plan_t, plan_v) - v_ego
    slope = float(np.min(dv / PLAN_SLOPE_T))
    a_plan_min = float(np.min(np.interp(PLAN_ACCEL_T, accel_t, plan_a)))
    f_plan = np.interp(slope, c.plan_slope_bp_user if user_active else c.plan_slope_bp, [0.0, 1.0])

    gap = lead_drel / max(v_ego, 1.0)
    f_lead = 1.0
    if lead_present:
      f_lead = np.interp(gap, c.lead_gap_bp, [0.0, 1.0])
      if lead_vlead < v_ego - 0.5:
        f_lead = min(f_lead, np.interp(gap, c.lead_gap_closing_bp, [0.0, 1.0]))
    if model_lead_prob > c.model_lead_prob and model_lead_x < c.model_lead_gap * max(v_ego, 1.0):
      f_lead = 0.0

    p_desire = max(max(desire_state[i] for i in TURN_KEEP_DESIRES),
                   max(desire_pred[h * DESIRE_WIDTH + i] for h in range(len(desire_pred) // DESIRE_WIDTH) for i in TURN_KEEP_DESIRES))
    f_desire = np.interp(p_desire, c.desire_bp, [1.0, 0.0])
    f_brake = np.interp(brake_prob, c.brake_prob_bp, [1.0, 0.0])
    f_blinker = np.interp(v_ego, c.blinker_speed_bp, [0.0, 1.0]) if blinker else 1.0
    f_speed = np.interp(v_ego, c.speed_bp, [0.0, 1.0])

    hard_veto = a_plan_min < c.plan_accel_veto or (lead_present and gap < c.lead_gap_veto)
    f = float(f_plan * f_lead * f_desire * f_brake * f_blinker * f_speed)
    self.debug.update(slope=slope, a_plan_min=a_plan_min, f_plan=f_plan, f_lead=f_lead, f_desire=f_desire,
                      f_brake=f_brake, f_blinker=f_blinker, f_speed=f_speed)
    return f, hard_veto

  def _curve_speed(self, a_lat_max, plan_t, plan_v, yaw_t, yaw_rate, x_t, plan_x):
    v = np.maximum(np.interp(CURVE_T, plan_t, plan_v), 1.0)
    kappa = np.abs(np.interp(CURVE_T, yaw_t, yaw_rate)) / v
    s = np.maximum(np.interp(CURVE_T, x_t, plan_x), 0.0)
    return float(np.min(np.sqrt(a_lat_max / np.maximum(kappa, 1e-5) + 2.0 * self.c.curve_decel * s)))

  def update(self, dt, enabled, gas_pressed, standstill, v_ego, v_cruise, personality, blinker,
             a_e2e, e2e_should_stop, a_mpc, a_cruise, lead_present, lead_drel, lead_vlead,
             plan_t, plan_v, accel_t, plan_a, x_t, plan_x, yaw_t, yaw_rate,
             desire_state, desire_pred, brake_prob, model_lead_prob, model_lead_x):
    c = self.c
    p = PERSONALITY_PARAMS.get(int(getattr(personality, 'raw', personality)), PERSONALITY_PARAMS[int(Personality.standard)])

    if not enabled:
      self.lift = 0.0
      self.output_lift = 0.0
      self.v_user = 0.0
      self.v_pre_lead = 0.0
      self.prev_enabled = False
      self.prev_gas = gas_pressed
      self.prev_v_cruise = v_cruise
      return a_e2e

    # user intent
    if not self.prev_enabled:
      self.anchor = v_ego
      self.v_user = v_ego
    if self.prev_gas and not gas_pressed:
      self.v_user = max(self.v_user, v_ego)
      self.anchor = max(self.anchor, v_ego)
    if self.prev_enabled and v_cruise > self.prev_v_cruise + 0.1 and not gas_pressed:
      self.v_user = max(self.v_user, v_ego) + c.set_speed_gain * (v_cruise - self.prev_v_cruise)

    # resume the speed a lead slowed us from, once it is gone
    lead_limited = lead_present and a_mpc < min(a_e2e + self.lift, a_cruise) - 0.05
    if lead_limited and not self.prev_lead_limited:
      self.v_pre_lead = v_ego
    lead_clear = not lead_present or lead_drel / max(v_ego, 1.0) > c.lead_clear_gap
    if self.v_pre_lead > 0.0 and not lead_limited and lead_clear:
      self.v_user = max(self.v_user, self.v_pre_lead)
      self.v_pre_lead = 0.0

    self.v_user = min(self.v_user, v_cruise) - c.v_user_decay * dt
    if v_ego < self.v_user - c.v_user_clear_drop:
      self.v_user = 0.0
    user_active = v_ego < self.v_user

    f, hard_veto = self._freedom(v_ego, plan_t, plan_v, accel_t, plan_a, desire_state, desire_pred, brake_prob, blinker,
                                 lead_present, lead_drel, lead_vlead, model_lead_prob, model_lead_x, user_active)
    hard_veto = hard_veto or gas_pressed or standstill or e2e_should_stop or a_e2e < c.e2e_accel_veto

    v_curve = self._curve_speed(p.a_lat_max, plan_t, plan_v, yaw_t, yaw_rate, x_t, plan_x)
    v_headroom = self.anchor * (1.0 + p.headroom_ratio) + p.headroom_abs
    v_goal = min(v_cruise, v_curve, max(v_headroom, self.v_user))
    a_track = float(np.clip(c.k_v * (v_goal - v_ego), 0.0, p.a_lift_max))
    target = max(0.0, a_track - a_e2e) * f

    if hard_veto:
      self.lift = 0.0
    elif target > self.lift:
      self.lift = min(target, self.lift + p.jerk_up * dt)
    else:
      self.lift = max(target, self.lift - c.jerk_down * dt)

    # the anchor is the speed the model chose by itself
    if self.lift < 0.05:
      self.anchor = v_ego
    else:
      self.anchor = min(self.anchor, v_ego) + max(a_e2e, 0.0) * dt

    self.prev_enabled = True
    self.prev_gas = gas_pressed
    self.prev_v_cruise = v_cruise
    self.prev_lead_limited = lead_limited
    a_out = a_e2e + self.lift
    if a_e2e < 0.0 and not c.allow_flip:
      a_out = min(a_out, 0.0)
    self.output_lift = a_out - a_e2e
    self.debug.update(lift=self.output_lift, target=target, f=f, hard_veto=float(hard_veto), v_goal=v_goal, v_curve=v_curve,
                      v_headroom=v_headroom, v_user=self.v_user, anchor=self.anchor, a_track=a_track)
    return a_out
