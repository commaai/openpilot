# Utilities for sanitizing routes of only essential data for testing car ports and doing validation.

from openpilot.tools.lib.logreader import LogIterable

PRESERVE_SERVICES = ["can", "carParams", "pandaStates", "pandaStateDEPRECATED"]

def sanitize(lr: LogIterable) -> LogIterable:
  return filter(lambda msg: msg.which() in PRESERVE_SERVICES, lr)
