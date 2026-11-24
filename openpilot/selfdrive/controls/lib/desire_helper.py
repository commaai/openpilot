from openpilot.cereal import log
from openpilot.common.constants import CV
from openpilot.common.realtime import DT_MDL

LaneChangeState = log.LaneChangeState
LaneChangeDirection = log.LaneChangeDirection

LANE_CHANGE_SPEED_MIN = 20 * CV.MPH_TO_MS
LANE_CHANGE_TIME_MAX = 10.
LANE_CHANGE_START_TIME = 0.5
TURN_ARM_SPEED = 20 * CV.MPH_TO_MS  # below lane change speed
TURN_PRED_THRESHOLD = 0.5  # model's own turn prediction for the blinker side
TURN_MOVING_SPEED = 0.3  # m/s
TURN_PULL_AWAY_ACCEL = 0.2  # m/s^2
TURN_PULSE_SPEED_UP = 1 * CV.MPH_TO_MS  # above the lowest speed since armed

class DesireHelper:
  def __init__(self):
    self.lane_change_state = LaneChangeState.off
    self.lane_change_direction = LaneChangeDirection.none
    self.lane_change_timer = 0.0
    self.prev_one_blinker = False
    self.desire = log.Desire.none
    self.turn_armed = LaneChangeDirection.none
    self.turn_min_speed = 0.0
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

    # experiment: blinker on while stopped arms a turn, pulse the turn desire once when pulling away
    if not one_blinker or not carstate.cruiseState.enabled:
      self.turn_armed = LaneChangeDirection.none
    elif self.turn_armed == LaneChangeDirection.none and v_ego < TURN_ARM_SPEED and self.turn_direction == LaneChangeDirection.none:
      self.turn_armed = self.get_lane_change_direction(carstate)
      self.turn_min_speed = v_ego
    elif self.turn_armed != LaneChangeDirection.none:
      self.turn_armed = self.get_lane_change_direction(carstate)
      self.turn_min_speed = min(self.turn_min_speed, v_ego)

    # the turn lasts until the model's turn desire prob falls, like a lane change
    if self.turn_direction != LaneChangeDirection.none:
      self.turn_timer += DT_MDL
      if not lateral_active or self.turn_timer > LANE_CHANGE_TIME_MAX or (turn_prob < 0.02 and self.turn_timer >= LANE_CHANGE_START_TIME):
        self.turn_direction = LaneChangeDirection.none

    self.desire = log.Desire.none
    # fire when the model itself predicts the turn on the blinker side, once moving
    pred = turn_pred[0] if self.turn_armed == LaneChangeDirection.left else turn_pred[1]
    # as soon as the car starts to speed up out of the stop, the turn must start before it is going fast
    pulling_away = v_ego > TURN_MOVING_SPEED and carstate.aEgo > TURN_PULL_AWAY_ACCEL
    if self.turn_armed != LaneChangeDirection.none and pulling_away and pred > TURN_PRED_THRESHOLD:
      self.desire = log.Desire.turnLeft if self.turn_armed == LaneChangeDirection.left else log.Desire.turnRight
      self.turn_direction = self.turn_armed
      self.turn_timer = 0.0
      self.turn_armed = LaneChangeDirection.none
      return
    if self.lane_change_state == LaneChangeState.laneChangeStarting:
      if self.lane_change_direction == LaneChangeDirection.left:
        self.desire = log.Desire.laneChangeLeft
      elif self.lane_change_direction == LaneChangeDirection.right:
        self.desire = log.Desire.laneChangeRight
