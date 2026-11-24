from openpilot.cereal import log
from openpilot.common.constants import CV
from openpilot.common.realtime import DT_MDL

LaneChangeState = log.LaneChangeState
LaneChangeDirection = log.LaneChangeDirection

LANE_CHANGE_SPEED_MIN = 20 * CV.MPH_TO_MS
LANE_CHANGE_TIME_MAX = 10.
LANE_CHANGE_START_TIME = 0.5
TURN_ARM_SPEED = 25 * CV.MPH_TO_MS
TURN_PRED_THRESHOLD = 0.7  # model's own turn prediction for the blinker side, nearest horizon
TURN_MOVING_SPEED = 0.3  # m/s
TURN_ROLLING_SPEED = 2.0  # m/s, below this only pulse once the car speeds up out of a stop
TURN_PULL_AWAY_ACCEL = 0.2  # m/s^2
TURN_REARM_STOP_TIME = 1.5  # s stopped before a new pulse is allowed
TURN_RESET_TIME = 1.0  # s without blinker or cruise before a new blinker gets a new pulse (both flicker)

class DesireHelper:
  def __init__(self):
    self.lane_change_state = LaneChangeState.off
    self.lane_change_direction = LaneChangeDirection.none
    self.lane_change_timer = 0.0
    self.prev_one_blinker = False
    self.desire = log.Desire.none
    self.turn_armed = LaneChangeDirection.none
    self.turn_fired = False
    self.turn_confirmed = False
    self.turn_stop_timer = 0.0
    self.turn_off_timer = 0.0
    self.turn_direction = LaneChangeDirection.none
    self.turn_timer = 0.0

  @staticmethod
  def get_lane_change_direction(CS):
    return LaneChangeDirection.left if CS.leftBlinker else LaneChangeDirection.right

  def update(self, carstate, lateral_active, lane_change_prob, turn_prob=0.0, turn_pred=(0.0, 0.0)):
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
    stopped = v_ego < TURN_MOVING_SPEED
    self.turn_stop_timer = self.turn_stop_timer + DT_MDL if stopped else 0.0
    side_pred = turn_pred[0] if carstate.leftBlinker else turn_pred[1]
    turn_allowed = one_blinker and carstate.cruiseState.enabled
    self.turn_off_timer = 0.0 if turn_allowed else self.turn_off_timer + DT_MDL
    if self.turn_off_timer > TURN_RESET_TIME:
      self.turn_fired = False
      self.turn_confirmed = False
    if not turn_allowed:
      self.turn_armed = LaneChangeDirection.none
    else:
      # the model saw this turn at some point with the blinker on
      self.turn_confirmed = self.turn_confirmed or side_pred > TURN_PRED_THRESHOLD
      if self.turn_fired and self.turn_stop_timer > TURN_REARM_STOP_TIME:
        self.turn_fired = False
        self.turn_direction = LaneChangeDirection.none
      lane_changing = self.lane_change_state in (LaneChangeState.laneChangeStarting, LaneChangeState.laneChangeFinishing)
      if v_ego < TURN_ARM_SPEED and not self.turn_fired and self.turn_direction == LaneChangeDirection.none and not lane_changing:
        self.turn_armed = self.get_lane_change_direction(carstate)
      else:
        self.turn_armed = LaneChangeDirection.none

    # the turn lasts until the model's turn desire prob falls, like a lane change
    if self.turn_direction != LaneChangeDirection.none:
      self.turn_timer += DT_MDL
      if not lateral_active or self.turn_timer > LANE_CHANGE_TIME_MAX or (turn_prob < 0.02 and self.turn_timer >= LANE_CHANGE_START_TIME):
        self.turn_direction = LaneChangeDirection.none

    self.desire = log.Desire.none
    # rolling: pulse when the model predicts the turn. out of a stop: pulse as soon as the car speeds up if the model
    # predicted the turn at any point, so the turn starts before the car is going fast
    rolling = v_ego > TURN_ROLLING_SPEED and side_pred > TURN_PRED_THRESHOLD
    pulling_away = not stopped and v_ego <= TURN_ROLLING_SPEED and carstate.aEgo > TURN_PULL_AWAY_ACCEL and self.turn_confirmed
    if self.turn_armed != LaneChangeDirection.none and (rolling or pulling_away):
      self.desire = log.Desire.turnLeft if self.turn_armed == LaneChangeDirection.left else log.Desire.turnRight
      self.turn_direction = self.turn_armed
      self.turn_timer = 0.0
      self.turn_armed = LaneChangeDirection.none
      self.turn_fired = True
      return
    if self.lane_change_state == LaneChangeState.laneChangeStarting:
      if self.lane_change_direction == LaneChangeDirection.left:
        self.desire = log.Desire.laneChangeLeft
      elif self.lane_change_direction == LaneChangeDirection.right:
        self.desire = log.Desire.laneChangeRight
