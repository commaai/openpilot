from common.op_params import opParams
from common.travis_checker import travis
from cereal import log

LaneChangeState = log.PathPlan.LaneChangeState
LaneChangeDirection = log.PathPlan.LaneChangeDirection


class LaneHugging:
  def __init__(self):
    self.op_params = opParams()
    self.direction = self.op_params.get('lane_hug_direction', None)  # if lane hugging is present and which side. None, 'left', or 'right'
    if isinstance(self.direction, str):
      self.direction = self.direction.lower()
    self.angle_offset = abs(self.op_params.get('lane_hug_angle_offset', 0.0))

  def modify_offset(self, angle_offset_average, lane_change_direction, lane_change_state):
    # negative angles: right
    # positive angles: left
    if not travis:
      starting = LaneChangeState.laneChangeStarting
      if self.direction == 'left' and ((lane_change_state == starting and lane_change_direction != LaneChangeDirection.left) or lane_change_state != starting):
        angle_offset_average = -self.angle_offset  # todo: not totally sure if this needs to be negative or positive lolol, will test and update
      elif self.direction == 'right' and ((lane_change_state == starting and lane_change_direction != LaneChangeDirection.right) or lane_change_state != starting):
        angle_offset_average = self.angle_offset
    return angle_offset_average
