# Utilities for sanitizing routes of only essential data for testing car ports and doing validation.

from openpilot.tools.lib.logreader import LogIterable, LogMessage


def sanitize_vin(vin: str):
  # (last 6 digits of vin are serial number https://en.wikipedia.org/wiki/Vehicle_identification_number)
  VIN_SENSITIVE = 6
  return vin[:-VIN_SENSITIVE] + "X" * VIN_SENSITIVE


def sanitize_msg(msg: LogMessage) -> LogMessage:
  if msg.which() == "carParams":
    msg = msg.as_builder()
    msg.carParams.carVin = sanitize_vin(msg.carParams.carVin)
    msg = msg.as_reader()
  return msg


PRESERVE_SERVICES = ["can", "carParams", "pandaStates", "pandaStateDEPRECATED"]


def sanitize(lr: LogIterable) -> LogIterable:
  filtered = filter(lambda msg: msg.which() in PRESERVE_SERVICES, lr)
  sanitized = map(sanitize_msg, filtered)
  return sanitized
