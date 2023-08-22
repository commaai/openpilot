import os

from cereal import car
from openpilot.common.params import Params
from openpilot.system.hardware import PC, TICI
from openpilot.selfdrive.manager.process import PythonProcess, NativeProcess, DaemonProcess, enabled_callback, disabled_callback

WEBCAM = os.getenv("USE_WEBCAM") is not None

def driverview(params: Params, CP: car.CarParams) -> bool:
  return params.get_bool("IsDriverViewEnabled")  # type: ignore

def notcar(params: Params, CP: car.CarParams) -> bool:
  return CP.notCar  # type: ignore

def iscar(params: Params, CP: car.CarParams) -> bool:
  return not CP.notCar

def logging(params, CP: car.CarParams) -> bool:
  return (not CP.notCar) or not params.get_bool("DisableLogging")

def ublox_available() -> bool:
  return os.path.exists('/dev/ttyHS0') and not os.path.exists('/persist/comma/use-quectel-gps')

def ublox(params, CP: car.CarParams) -> bool:
  use_ublox = ublox_available()
  if use_ublox != params.get_bool("UbloxAvailable"):
    params.put_bool("UbloxAvailable", use_ublox)
  return use_ublox

def qcomgps(params, CP: car.CarParams) -> bool:
  return ublox_available()

procs = [
  NativeProcess("camerad", "system/camerad", ["./camerad"], offroad=driverview),
  NativeProcess("clocksd", "system/clocksd", ["./clocksd"]),
  NativeProcess("logcatd", "system/logcatd", ["./logcatd"]),
  NativeProcess("proclogd", "system/proclogd", ["./proclogd"]),
  PythonProcess("logmessaged", "system.logmessaged", offroad=enabled_callback),
  PythonProcess("micd", "system.micd", onroad=iscar),
  PythonProcess("timezoned", "system.timezoned", enabled=not PC, offroad=enabled_callback),

  DaemonProcess("manage_athenad", "selfdrive.athena.manage_athenad", "AthenadPid"),
  NativeProcess("dmonitoringmodeld", "selfdrive/modeld", ["./dmonitoringmodeld"], enabled=(not PC or WEBCAM), offroad=driverview),
  NativeProcess("encoderd", "system/loggerd", ["./encoderd"]),
  NativeProcess("stream_encoderd", "system/loggerd", ["./encoderd", "--stream"], onroad=notcar),
  NativeProcess("loggerd", "system/loggerd", ["./loggerd"], onroad=logging),
  NativeProcess("modeld", "selfdrive/modeld", ["./modeld"]),
  NativeProcess("mapsd", "selfdrive/navd", ["./mapsd"]),
  NativeProcess("navmodeld", "selfdrive/modeld", ["./navmodeld"]),
  NativeProcess("sensord", "system/sensord", ["./sensord"], enabled=not PC),
  NativeProcess("ui", "selfdrive/ui", ["./ui"], offroad=enabled_callback, watchdog_max_dt=(5 if not PC else None)),
  NativeProcess("soundd", "selfdrive/ui/soundd", ["./soundd"]),
  NativeProcess("locationd", "selfdrive/locationd", ["./locationd"]),
  NativeProcess("boardd", "selfdrive/boardd", ["./boardd"], enabled=False),
  PythonProcess("calibrationd", "selfdrive.locationd.calibrationd"),
  PythonProcess("torqued", "selfdrive.locationd.torqued"),
  PythonProcess("controlsd", "selfdrive.controls.controlsd"),
  PythonProcess("deleter", "system.loggerd.deleter", offroad=enabled_callback),
  PythonProcess("dmonitoringd", "selfdrive.monitoring.dmonitoringd", enabled=(not PC or WEBCAM), offroad=driverview),
  PythonProcess("laikad", "selfdrive.locationd.laikad"),
  PythonProcess("rawgpsd", "system.sensord.rawgps.rawgpsd", enabled=TICI, onroad=qcomgps),
  PythonProcess("navd", "selfdrive.navd.navd"),
  PythonProcess("pandad", "selfdrive.boardd.pandad", offroad=enabled_callback),
  PythonProcess("paramsd", "selfdrive.locationd.paramsd"),
  NativeProcess("ubloxd", "system/ubloxd", ["./ubloxd"], enabled=TICI, onroad=ublox),
  PythonProcess("pigeond", "system.sensord.pigeond", enabled=TICI, onroad=ublox),
  PythonProcess("plannerd", "selfdrive.controls.plannerd"),
  PythonProcess("radard", "selfdrive.controls.radard"),
  PythonProcess("thermald", "selfdrive.thermald.thermald", offroad=enabled_callback),
  PythonProcess("tombstoned", "selfdrive.tombstoned", enabled=not PC, offroad=enabled_callback),
  PythonProcess("updated", "selfdrive.updated", enabled=not PC, onroad=disabled_callback, offroad=enabled_callback),
  PythonProcess("uploader", "system.loggerd.uploader", offroad=enabled_callback),
  PythonProcess("statsd", "selfdrive.statsd", offroad=enabled_callback),

  # debug procs
  NativeProcess("bridge", "cereal/messaging", ["./bridge"], onroad=notcar),
  PythonProcess("webjoystick", "tools.bodyteleop.web", onroad=notcar),
]

managed_processes = {p.name: p for p in procs}
