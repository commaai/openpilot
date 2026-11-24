from openpilot.cereal import log
from openpilot.common.constants import CV
from openpilot.common.realtime import DT_MDL

LaneChangeState = log.LaneChangeState
LaneChangeDirection = log.LaneChangeDirection

LANE_CHANGE_SPEED_MIN = 20 * CV.MPH_TO_MS
LANE_CHANGE_TIME_MAX = 10.
LANE_CHANGE_START_TIME = 0.5
TURN_ARM_SPEED = 40 * CV.MPH_TO_MS
TURN_POSSIBLE = 0.2  # model's turn prediction for the blinker side, any horizon, last 2 s: the turn is possible
TURN_LIKELY = 0.5  # pulse even when the plan does not speed up, below TURN_LIKELY_SPEED
TURN_LIKELY_SPEED = 25 * CV.MPH_TO_MS
TURN_PULL_AWAY_ACCEL = 0.2  # m/s^2, the car starts to move out of a stop
TURN_PLAN_SPEEDUP = 0.3  # m/s the plan gains in 2 s: the car is about to speed up, out of a stop or out of a slow turn
TURN_POSSIBLE_DECAY = 0.5  # 1/s, holds the prediction peak for 2 s
TURN_ROLLING_SPEED = 2.0  # m/s, below this the blinker alone is enough
TURN_REARM_STOP_TIME = 1.0  # s below TURN_ROLLING_SPEED (a stop or a rolling stop) before a new pulse is allowed
KEEP_PRED_THRESHOLD = 0.5  # model's keep left/right prediction for the blinker side, e.g. a new lane opening
TURN_RESET_TIME = 1.0  # s without blinker or cruise before a new blinker gets a new pulse (both flicker)
TURN_BLINKER_HOLD = 3.0  # s the turn stays armed after the blinker goes off, it can cancel right at the pull-away

class DesireHelper:
  def __init__(self):
    self.lane_change_state = LaneChangeState.off
    self.lane_change_direction = LaneChangeDirection.none
    self.lane_change_timer = 0.0
    self.prev_one_blinker = False
    self.desire = log.Desire.none
    self.turn_armed = LaneChangeDirection.none
    self.turn_fired = False
    self.turn_possible = 0.0
    self.prev_plan_speedup = True
    self.turn_moved = False
    self.turn_blinker = LaneChangeDirection.none
    self.turn_blinker_timer = 0.0
    self.keep_fired = False
    self.turn_stop_timer = 0.0
    self.turn_off_timer = 0.0
    self.turn_direction = LaneChangeDirection.none
    self.turn_timer = 0.0

  @staticmethod
  def get_lane_change_direction(CS):
    return LaneChangeDirection.left if CS.leftBlinker else LaneChangeDirection.right

  def update(self, carstate, lateral_active, lane_change_prob, turn_prob=0.0, turn_pred=(0.0, 0.0), plan_speedup=0.0, keep_pred=(0.0, 0.0)):
    v_ego = carstate.vEgo
    one_blinker = carstate.leftBlinker != carstate.rightBlinker
    below_lane_change_speed = v_ego < LANE_CHANGE_SPEED_MIN

    if not lateral_active or self.lane_change_timer > LANE_CHANGE_TIME_MAX:
      self.lane_change_state = LaneChangeState.off
      self.lane_change_direction = LaneChangeDirection.none
      self.lane_change_timer = 0.0
    else:
      if self.lane_change_state == LaneChangeState.off and one_blinker and not self.prev_one_blinker and not below_lane_change_speed:
        self.lane_change_state = LaneChangeState.preLaneChange
        self.lane_change_timer = 0.0
        # Initialize lane change direction to prevent UI alert flicker
        self.lane_change_direction = self.get_lane_change_direction(carstate)

      elif self.lane_change_state == LaneChangeState.preLaneChange:
        # Update lane change direction
        self.lane_change_direction = self.get_lane_change_direction(carstate)

        torque_applied = carstate.steeringPressed and \
                         ((carstate.steeringTorque > 0 and self.lane_change_direction == LaneChangeDirection.left) or
                          (carstate.steeringTorque < 0 and self.lane_change_direction == LaneChangeDirection.right))

        blindspot_detected = ((carstate.leftBlindspot and self.lane_change_direction == LaneChangeDirection.left) or
                              (carstate.rightBlindspot and self.lane_change_direction == LaneChangeDirection.right))

        if not one_blinker or below_lane_change_speed:
          self.lane_change_state = LaneChangeState.off
          self.lane_change_direction = LaneChangeDirection.none
          self.lane_change_timer = 0.0
        elif torque_applied and not blindspot_detected:
          self.lane_change_state = LaneChangeState.laneChangeStarting
          self.lane_change_timer = 0.0

      elif self.lane_change_state == LaneChangeState.laneChangeStarting:
        self.lane_change_timer += DT_MDL

        if lane_change_prob < 0.02 and self.lane_change_timer >= LANE_CHANGE_START_TIME:
          self.lane_change_timer = 0.0
          if one_blinker:
            self.lane_change_state = LaneChangeState.preLaneChange
            self.lane_change_direction = self.get_lane_change_direction(carstate)
          else:
            self.lane_change_state = LaneChangeState.off
            self.lane_change_direction = LaneChangeDirection.none

    self.prev_one_blinker = one_blinker and lateral_active

    # experiment: with the blinker on below 25 mph, pulse the turn desire once when the model itself predicts that turn.
    # one pulse per blinker, but a stop allows a new one: the model can see the turn early, then stop for a light first
    self.turn_stop_timer = self.turn_stop_timer + DT_MDL if v_ego < TURN_ROLLING_SPEED else 0.0
    speeding_up = plan_speedup > TURN_PLAN_SPEEDUP
    speedup_edge = speeding_up and not self.prev_plan_speedup
    self.prev_plan_speedup = speeding_up
    # keep the last blinker direction for a moment after it goes off
    if one_blinker:
      self.turn_blinker = self.get_lane_change_direction(carstate)
      self.turn_blinker_timer = 0.0
    else:
      self.turn_blinker_timer += DT_MDL
      if self.turn_blinker_timer > TURN_BLINKER_HOLD:
        self.turn_blinker = LaneChangeDirection.none
    blinker_on = self.turn_blinker != LaneChangeDirection.none
    side_pred = turn_pred[0] if self.turn_blinker == LaneChangeDirection.left else turn_pred[1]
    self.turn_possible = max(side_pred, self.turn_possible - TURN_POSSIBLE_DECAY * DT_MDL)
    turn_allowed = blinker_on and carstate.cruiseState.enabled
    self.turn_off_timer = 0.0 if turn_allowed else self.turn_off_timer + DT_MDL
    if self.turn_off_timer > TURN_RESET_TIME:
      self.turn_fired = False
      self.keep_fired = False
    if not turn_allowed:
      self.turn_armed = LaneChangeDirection.none
    else:
      # a new stop after driving on allows a new pulse, e.g. a pulse before a red light, then the turn after it
      self.turn_moved = self.turn_moved or v_ego > TURN_ROLLING_SPEED
      if self.turn_fired and self.turn_moved and self.turn_stop_timer > TURN_REARM_STOP_TIME:
        self.turn_fired = False
        self.turn_direction = LaneChangeDirection.none
      lane_changing = self.lane_change_state in (LaneChangeState.laneChangeStarting, LaneChangeState.laneChangeFinishing)
      if v_ego < TURN_ARM_SPEED and not self.turn_fired and self.turn_direction == LaneChangeDirection.none and not lane_changing:
        self.turn_armed = self.turn_blinker
      else:
        self.turn_armed = LaneChangeDirection.none

    # the turn lasts until the model's turn desire prob falls, like a lane change
    if self.turn_direction != LaneChangeDirection.none:
      self.turn_timer += DT_MDL
      if not lateral_active or self.turn_timer > LANE_CHANGE_TIME_MAX or (turn_prob < 0.02 and self.turn_timer >= LANE_CHANGE_START_TIME):
        self.turn_direction = LaneChangeDirection.none

    self.desire = log.Desire.none
    # pulse when the model's plan is about to speed up: out of a stop (blinker alone is enough) or out of the slow part
    # of a turn (the model must see the turn as possible). a likely turn pulses without waiting for the plan
    at_stop = v_ego < TURN_ROLLING_SPEED
    # at a stop the plan flickers while waiting, so also wait for the car to start to move
    pull_away = at_stop and speeding_up and carstate.aEgo > TURN_PULL_AWAY_ACCEL
    rolling = not at_stop and speedup_edge and self.turn_possible > TURN_POSSIBLE
    likely = not at_stop and v_ego < TURN_LIKELY_SPEED and self.turn_possible > TURN_LIKELY
    if self.turn_armed != LaneChangeDirection.none and (pull_away or rolling or likely):
      self.desire = log.Desire.turnLeft if self.turn_armed == LaneChangeDirection.left else log.Desire.turnRight
      self.turn_direction = self.turn_armed
      self.turn_timer = 0.0
      self.turn_armed = LaneChangeDirection.none
      self.turn_fired = True
      self.turn_moved = False
      return
    # keep left/right: one pulse per blinker when the model itself predicts it on that side, e.g. a new lane opening
    keep_side_pred = keep_pred[0] if self.turn_blinker == LaneChangeDirection.left else keep_pred[1]
    lane_changing = self.lane_change_state in (LaneChangeState.laneChangeStarting, LaneChangeState.laneChangeFinishing)
    if turn_allowed and not self.keep_fired and not lane_changing and self.turn_direction == LaneChangeDirection.none and \
       keep_side_pred > KEEP_PRED_THRESHOLD:
      self.desire = log.Desire.keepLeft if self.turn_blinker == LaneChangeDirection.left else log.Desire.keepRight
      self.keep_fired = True
      return
    if self.lane_change_state == LaneChangeState.laneChangeStarting:
      if self.lane_change_direction == LaneChangeDirection.left:
        self.desire = log.Desire.laneChangeLeft
      elif self.lane_change_direction == LaneChangeDirection.right:
        self.desire = log.Desire.laneChangeRight
