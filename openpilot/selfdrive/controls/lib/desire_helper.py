from openpilot.cereal import log

class DesireHelper:
  def __init__(self):
    self.lane_change_state = log.LaneChangeState.off
    self.lane_change_direction = log.LaneChangeDirection.none
    self.desire = log.Desire.none
    self.turn_pulsed = False

  def update(self, carstate, lateral_active):
    if not lateral_active or carstate.leftBlinker == carstate.rightBlinker:
      self.desire = log.Desire.none
      self.turn_pulsed = False
      return

    turn = log.Desire.turnLeft if carstate.leftBlinker else log.Desire.turnRight

    if self.turn_pulsed:
      self.desire = turn
      return

    blindspot_detected = carstate.leftBlindspot if carstate.leftBlinker else carstate.rightBlindspot
    if blindspot_detected:
      self.desire = log.Desire.none
      return

    self.desire = turn
    self.turn_pulsed = True
