# Utilities for sanetizing routes of only essential data

from openpilot.tools.lib.logreader import LogIterable

PRESERVE_SERVICES = ["can", "carParams", "pandaStates", "pandaStateDEPRECATED"]

def sanetize(lr: LogIterable) -> LogIterable:
  return filter(lambda msg: msg.which() in PRESERVE_SERVICES, lr)
