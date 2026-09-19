import time

from openpilot.cereal import log

TURN_PULSE_INTERVAL = 1.0  # seconds between normal-sized turn pulses


class DesireHelper:
  def __init__(self):
    # Turn requests are not lane changes; keep lane-change UI/control metadata off.
    self.lane_change_state = log.LaneChangeState.off
    self.lane_change_direction = log.LaneChangeDirection.none
    self.desire = log.Desire.none
    self.last_pulse_time = None
    self.last_turn = log.Desire.none

  def update(self, carstate, lateral_active):
    previous_desire = self.desire
    self.desire = log.Desire.none

    if not lateral_active or carstate.leftBlinker == carstate.rightBlinker:
      self.last_pulse_time = None
      self.last_turn = log.Desire.none
      return
    blindspot_detected = carstate.leftBlindspot if carstate.leftBlinker else carstate.rightBlindspot
    if blindspot_detected:
      self.last_pulse_time = None
      self.last_turn = log.Desire.none
      return

    turn = log.Desire.turnLeft if carstate.leftBlinker else log.Desire.turnRight
    # Even after a delayed update, provide a low sample to re-arm modeld's edge detector.
    if previous_desire == turn:
      return
    now = time.monotonic()
    if self.last_pulse_time is None or turn != self.last_turn or now - self.last_pulse_time >= TURN_PULSE_INTERVAL:
      # One update high, then none until the next pulse. modeld still receives
      # only 0/1 inputs. Stopping pulses does not cancel the model's maneuver.
      self.desire = turn
      self.last_pulse_time = now
      self.last_turn = turn
