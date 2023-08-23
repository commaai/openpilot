import os

from cereal import car
from openpilot.common.params import Params
from openpilot.system.hardware import PC, TICI
from openpilot.selfdrive.manager.process import PythonProcess, NativeProcess, DaemonProcess

WEBCAM = os.getenv("USE_WEBCAM") is not None

def driverview(started: bool, params: Params, CP: car.CarParams) -> bool:
  return started or params.get_bool("IsDriverViewEnabled")

def notcar(started: bool, params: Params, CP: car.CarParams) -> bool:
  return CP.notCar  # type: ignore

def iscar(started: bool, params: Params, CP: car.CarParams) -> bool:
  return not CP.notCar

def logging(started, params, CP: car.CarParams) -> bool:
  run = (not CP.notCar) or not params.get_bool("DisableLogging")
  return started and run

def ublox_available() -> bool:
  return os.path.exists('/dev/ttyHS0') and not os.path.exists('/persist/comma/use-quectel-gps')

def ublox(started, params, CP: car.CarParams) -> bool:
  use_ublox = ublox_available()
  if use_ublox != params.get_bool("UbloxAvailable"):
    params.put_bool("UbloxAvailable", use_ublox)
  return started and use_ublox

def qcomgps(started, params, CP: car.CarParams) -> bool:
  return started and not ublox_available()

def always(started, params, CP: car.CarParams) -> bool:
  return True

def only_offroad(started, params, CP: car.CarParams) -> bool:
  return not started

procs = [
  NativeProcess("camerad", "system/camerad", ["./camerad"], callback=driverview),
  NativeProcess("clocksd", "system/clocksd", ["./clocksd"]),
  NativeProcess("logcatd", "system/logcatd", ["./logcatd"]),
  NativeProcess("proclogd", "system/proclogd", ["./proclogd"]),
  PythonProcess("logmessaged", "system.logmessaged", callback=always),
  PythonProcess("micd", "system.micd", callback=iscar),
  PythonProcess("timezoned", "system.timezoned", enabled=not PC, callback=always),

  DaemonProcess("manage_athenad", "selfdrive.athena.manage_athenad", "AthenadPid"),
  NativeProcess("dmonitoringmodeld", "selfdrive/modeld", ["./dmonitoringmodeld"], enabled=(not PC or WEBCAM), callback=driverview),
  NativeProcess("encoderd", "system/loggerd", ["./encoderd"]),
  NativeProcess("stream_encoderd", "system/loggerd", ["./encoderd", "--stream"], onroad=False, callback=notcar),
  NativeProcess("loggerd", "system/loggerd", ["./loggerd"], onroad=False, callback=logging),
  NativeProcess("modeld", "selfdrive/modeld", ["./modeld"]),
  NativeProcess("mapsd", "selfdrive/navd", ["./mapsd"]),
  NativeProcess("navmodeld", "selfdrive/modeld", ["./navmodeld"]),
  NativeProcess("sensord", "system/sensord", ["./sensord"], enabled=not PC),
  NativeProcess("ui", "selfdrive/ui", ["./ui"], callback=always, watchdog_max_dt=(5 if not PC else None)),
  NativeProcess("soundd", "selfdrive/ui/soundd", ["./soundd"]),
  NativeProcess("locationd", "selfdrive/locationd", ["./locationd"]),
  NativeProcess("boardd", "selfdrive/boardd", ["./boardd"], enabled=False),
  PythonProcess("calibrationd", "selfdrive.locationd.calibrationd"),
  PythonProcess("torqued", "selfdrive.locationd.torqued"),
  PythonProcess("controlsd", "selfdrive.controls.controlsd"),
  PythonProcess("deleter", "system.loggerd.deleter", callback=always),
  PythonProcess("dmonitoringd", "selfdrive.monitoring.dmonitoringd", enabled=(not PC or WEBCAM), callback=driverview),
  PythonProcess("laikad", "selfdrive.locationd.laikad"),
  PythonProcess("rawgpsd", "system.sensord.rawgps.rawgpsd", enabled=TICI, onroad=False, callback=qcomgps),
  PythonProcess("navd", "selfdrive.navd.navd"),
  PythonProcess("pandad", "selfdrive.boardd.pandad", callback=always),
  PythonProcess("paramsd", "selfdrive.locationd.paramsd"),
  NativeProcess("ubloxd", "system/ubloxd", ["./ubloxd"], enabled=TICI, onroad=False, callback=ublox),
  PythonProcess("pigeond", "system.sensord.pigeond", enabled=TICI, onroad=False, callback=ublox),
  PythonProcess("plannerd", "selfdrive.controls.plannerd"),
  PythonProcess("radard", "selfdrive.controls.radard"),
  PythonProcess("thermald", "selfdrive.thermald.thermald", callback=always),
  PythonProcess("tombstoned", "selfdrive.tombstoned", enabled=not PC, callback=always),
  PythonProcess("updated", "selfdrive.updated", enabled=not PC, callback=only_offroad),
  PythonProcess("uploader", "system.loggerd.uploader", callback=always),
  PythonProcess("statsd", "selfdrive.statsd", callback=always),

  # debug procs
  NativeProcess("bridge", "cereal/messaging", ["./bridge"], onroad=False, callback=notcar),
  PythonProcess("webjoystick", "tools.bodyteleop.web", onroad=False, callback=notcar),
]

managed_processes = {p.name: p for p in procs}
