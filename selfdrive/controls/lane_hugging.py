from common.op_params import opParams


class LaneHugging:
  def __init__(self):
    self.op_params = opParams()
    self.direction = self.op_params.get('lane_hug_direction', None)  # if lane hugging is present and which side. None, 'left', or 'right'
    if isinstance(self.direction, str):
      self.direction = self.direction.lower()
    self.angle_offset = abs(self.op_params.get('lane_hug_angle_offset', 0.0))

  def offset_mod(self, angle_steers_des):
    # negative angles: right
    # positive angles: left
    if self.direction == 'left':
      angle_steers_des -= self.angle_offset
    elif self.direction == 'right':
      angle_steers_des += self.angle_offset
    return angle_steers_des
