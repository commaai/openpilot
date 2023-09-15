import os

from cereal import car
from openpilot.common.params import Params
from openpilot.system.hardware import PC, TICI
from openpilot.selfdrive.manager.process import PythonProcess, NativeProcess, DaemonProcess

WEBCAM = os.getenv("USE_WEBCAM") is not None

def driverview(started: bool, params: Params, CP: car.CarParams) -> bool:
  return started or params.get_bool("IsDriverViewEnabled")

def notcar(started: bool, params: Params, CP: car.CarParams) -> bool:
  return started and CP.notCar

def iscar(started: bool, params: Params, CP: car.CarParams) -> bool:
  return started and not CP.notCar

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

def always_run(started, params, CP: car.CarParams) -> bool:
  return True

def only_onroad(started: bool, params, CP: car.CarParams) -> bool:
  return started

def only_offroad(started, params, CP: car.CarParams) -> bool:
  return not started

procs = [
  DaemonProcess("manage_athenad", "selfdrive.athena.manage_athenad", "AthenadPid"),

  NativeProcess("camerad", "system/camerad", ["./camerad"], driverview),
  NativeProcess("clocksd", "system/clocksd", ["./clocksd"], only_onroad),
  NativeProcess("logcatd", "system/logcatd", ["./logcatd"], only_onroad),
  NativeProcess("proclogd", "system/proclogd", ["./proclogd"], only_onroad),
  PythonProcess("logmessaged", "system.logmessaged", always_run),
  PythonProcess("micd", "system.micd", iscar),
  PythonProcess("timezoned", "system.timezoned", always_run, enabled=not PC),

  PythonProcess("dmonitoringmodeld", "selfdrive.modeld.dmonitoringmodeld", driverview, enabled=(not PC or WEBCAM)),
  NativeProcess("encoderd", "system/loggerd", ["./encoderd"], only_onroad),
  NativeProcess("stream_encoderd", "system/loggerd", ["./encoderd", "--stream"], notcar),
  NativeProcess("loggerd", "system/loggerd", ["./loggerd"], logging),
  NativeProcess("modeld", "selfdrive/modeld", ["./modeld"], only_onroad),
  NativeProcess("mapsd", "selfdrive/navd", ["./mapsd"], only_onroad),
  PythonProcess("navmodeld", "selfdrive.modeld.navmodeld", only_onroad),
  NativeProcess("sensord", "system/sensord", ["./sensord"], only_onroad, enabled=not PC),
  NativeProcess("ui", "selfdrive/ui", ["./ui"], always_run, watchdog_max_dt=(5 if not PC else None)),
  NativeProcess("soundd", "selfdrive/ui/soundd", ["./soundd"], only_onroad),
  NativeProcess("locationd", "selfdrive/locationd", ["./locationd"], only_onroad),
  NativeProcess("boardd", "selfdrive/boardd", ["./boardd"], always_run, enabled=False),
  PythonProcess("calibrationd", "selfdrive.locationd.calibrationd", only_onroad),
  PythonProcess("torqued", "selfdrive.locationd.torqued", only_onroad),
  PythonProcess("controlsd", "selfdrive.controls.controlsd", only_onroad),
  PythonProcess("deleter", "system.loggerd.deleter", always_run),
  PythonProcess("dmonitoringd", "selfdrive.monitoring.dmonitoringd", driverview, enabled=(not PC or WEBCAM)),
  PythonProcess("laikad", "selfdrive.locationd.laikad", only_onroad),
  PythonProcess("rawgpsd", "system.sensord.rawgps.rawgpsd", qcomgps, enabled=TICI),
  PythonProcess("navd", "selfdrive.navd.navd", only_onroad),
  PythonProcess("pandad", "selfdrive.boardd.pandad", always_run),
  PythonProcess("paramsd", "selfdrive.locationd.paramsd", only_onroad),
  NativeProcess("ubloxd", "system/ubloxd", ["./ubloxd"], ublox, enabled=TICI),
  PythonProcess("pigeond", "system.sensord.pigeond", ublox, enabled=TICI),
  PythonProcess("plannerd", "selfdrive.controls.plannerd", only_onroad),
  PythonProcess("radard", "selfdrive.controls.radard", only_onroad),
  PythonProcess("thermald", "selfdrive.thermald.thermald", always_run),
  PythonProcess("tombstoned", "selfdrive.tombstoned", always_run, enabled=not PC),
  PythonProcess("updated", "selfdrive.updated", only_offroad, enabled=not PC),
  PythonProcess("uploader", "system.loggerd.uploader", always_run),
  PythonProcess("statsd", "selfdrive.statsd", always_run),

  # debug procs
  NativeProcess("bridge", "cereal/messaging", ["./bridge"], notcar),
  PythonProcess("webjoystick", "tools.bodyteleop.web", notcar),
]

managed_processes = {p.name: p for p in procs}
