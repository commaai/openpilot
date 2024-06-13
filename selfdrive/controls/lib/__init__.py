from cereal import car
from openpilot.system.version import get_build_metadata

EventName = car.CarEvent.EventName


def get_startup_event(car_recognized, controller_available, fw_seen):
  build_metadata = get_build_metadata()
  if build_metadata.openpilot.comma_remote and build_metadata.tested_channel:
    event = EventName.startup
  else:
    event = EventName.startupMaster

  if not car_recognized:
    if fw_seen:
      event = EventName.startupNoCar
    else:
      event = EventName.startupNoFw
  elif car_recognized and not controller_available:
    event = EventName.startupNoControl
  return event
