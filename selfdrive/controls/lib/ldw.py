from cereal import log
from openpilot.common.realtime import DT_CTRL
from openpilot.common.conversions import Conversions as CV


CAMERA_OFFSET = 0.04
LDW_MIN_SPEED = 31 * CV.MPH_TO_MS
LANE_DEPARTURE_THRESHOLD = 0.1

class LaneDepartureWarning:
  def __init__(self):
    self.left = False
    self.right = False
    self.last_blinker_frame = 0

  def update(self, frame, modelV2, CS, CC):
    if CS.leftBlinker or CS.rightBlinker:
      self.last_blinker_frame = frame

    recent_blinker = (frame - self.last_blinker_frame) * DT_CTRL < 5.0  # 5s blinker cooldown
    ldw_allowed = CS.vEgo > LDW_MIN_SPEED and not recent_blinker and not CC.latActive

    desire_prediction = modelV2.meta.desirePrediction
    if len(desire_prediction) and ldw_allowed:
      right_lane_visible = modelV2.laneLineProbs[2] > 0.5
      left_lane_visible = modelV2.laneLineProbs[1] > 0.5
      l_lane_change_prob = desire_prediction[log.Desire.laneChangeLeft]
      r_lane_change_prob = desire_prediction[log.Desire.laneChangeRight]

      lane_lines = modelV2.laneLines
      l_lane_close = left_lane_visible and (lane_lines[1].y[0] > -(1.08 + CAMERA_OFFSET))
      r_lane_close = right_lane_visible and (lane_lines[2].y[0] < (1.08 - CAMERA_OFFSET))

      self.left = bool(l_lane_change_prob > LANE_DEPARTURE_THRESHOLD and l_lane_close)
      self.right = bool(r_lane_change_prob > LANE_DEPARTURE_THRESHOLD and r_lane_close)
    else:
      self.left, self.right = False, False

  @property
  def warning(self) -> bool:
    return bool(self.left or self.right)
