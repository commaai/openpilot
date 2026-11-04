from openpilot.common.params import Params


def get_gps_location_service(params: Params) -> str:
  if params.get_bool("UbloxAvailable"):
    return "gpsLocationExternal"
  else:
    return "gpsLocation"
