import math

from openpilot.cereal import log

NAV_MAX_SPEED = 8.3                      # m/s
DESIRE_ON_BEARING = math.radians(15.)
DESIRE_OFF_BEARING = math.radians(10.)


class NavDesireInjector:
  """Maps the parkNavSignal (rotation toward a GPS destination) onto the model's
  trained turn desires (log.Desire.turnLeft / turnRight), reusing the same channel
  the blinker path uses. The physical blinker desire always takes precedence.

  Turn activation/deactivation uses a hysteresis band around the relBearing
  threshold. relBearing is in the NED convention: positive = destination to the right.
  """

  def __init__(self):
    self.turn_active = False

  def update(self, nav_signal, nav_alive: bool, CS, lat_active: bool,
             v_ego: float, blinker_desire: log.Desire) -> log.Desire:
    if blinker_desire != log.Desire.none:
      self.turn_active = False
      return blinker_desire

    nav_ok = nav_alive and nav_signal.valid and lat_active and v_ego < NAV_MAX_SPEED
    if not nav_ok:
      self.turn_active = False
      return log.Desire.none

    abs_bearing = abs(nav_signal.relBearing)
    if not self.turn_active:
      if abs_bearing > DESIRE_ON_BEARING:
        self.turn_active = True
      else:
        return log.Desire.none
    elif abs_bearing < DESIRE_OFF_BEARING:
      self.turn_active = False
      return log.Desire.none
    # else: inside the hysteresis band, stay active

    turn = log.Desire.turnRight if nav_signal.relBearing > 0 else log.Desire.turnLeft
    blindspot = CS.rightBlindspot if turn == log.Desire.turnRight else CS.leftBlindspot
    if blindspot:
      return log.Desire.none
    return turn
