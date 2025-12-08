from cereal import log


class IgnitionState:
  def __init__(self):
    self.ignition_can_seen = False

  def update(self, panda_states) -> bool:
    valid_states = [ps for ps in panda_states if ps.pandaType != log.PandaState.PandaType.unknown]
    if not valid_states:
      return False
    # Prefer CAN ignition once detected to avoid false positives from ignitionLine,
    # which can remain high after the car is turned off on some vehicles
    if any(ps.ignitionCan for ps in valid_states):
      self.ignition_can_seen = True
      return True

    return False if self.ignition_can_seen else any(ps.ignitionLine for ps in valid_states)

ignition_state = IgnitionState()
