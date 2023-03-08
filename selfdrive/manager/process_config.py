import os

from cereal import car
from common.params import Params
from system.hardware import PC, TICI
from selfdrive.manager.process import PythonProcess, NativeProcess, DaemonProcess

WEBCAM = os.getenv("USE_WEBCAM") is not None

def driverview(started: bool, params: Params, CP: car.CarParams) -> bool:
  return params.get_bool("IsDriverViewEnabled")  # type: ignore

def notcar(started: bool, params: Params, CP: car.CarParams) -> bool:
  return CP.notCar  # type: ignore

def logging(started, params, CP: car.CarParams) -> bool:
  run = (not CP.notCar) or not params.get_bool("DisableLogging")
  return started and run

def ublox(started, params, CP: car.CarParams) -> bool:
  use_ublox = os.path.exists('/dev/ttyHS0') and not os.path.exists('/persist/comma/use-quectel-gps')
  params.put_bool("UbloxAvailable", use_ublox)
  return started and use_ublox

procs = [
  # due to qualcomm kernel bugs SIGKILLing camerad sometimes causes page table corruption
  NativeProcess("camerad", "system/camerad", ["./camerad"], unkillable=True, callback=driverview),
  NativeProcess("clocksd", "system/clocksd", ["./clocksd"]),
  NativeProcess("logcatd", "system/logcatd", ["./logcatd"]),
  NativeProcess("proclogd", "system/proclogd", ["./proclogd"]),
  PythonProcess("logmessaged", "system.logmessaged", offroad=True),
  PythonProcess("micd", "system.micd"),
  PythonProcess("timezoned", "system.timezoned", enabled=not PC, offroad=True),

  DaemonProcess("manage_athenad", "selfdrive.athena.manage_athenad", "AthenadPid"),
  NativeProcess("dmonitoringmodeld", "selfdrive/modeld", ["./dmonitoringmodeld"], enabled=(not PC or WEBCAM), callback=driverview),
  NativeProcess("encoderd", "system/loggerd", ["./encoderd"]),
  NativeProcess("loggerd", "system/loggerd", ["./loggerd"], onroad=False, callback=logging),
  NativeProcess("modeld", "selfdrive/modeld", ["./modeld"]),
  NativeProcess("mapsd", "selfdrive/navd", ["./map_renderer"], enabled=False),
  NativeProcess("navmodeld", "selfdrive/modeld", ["./navmodeld"], enabled=False),
  NativeProcess("sensord", "system/sensord", ["./sensord"], enabled=not PC),
  NativeProcess("ui", "selfdrive/ui", ["./ui"], offroad=True, watchdog_max_dt=(5 if not PC else None)),
  NativeProcess("soundd", "selfdrive/ui/soundd", ["./soundd"], offroad=True),
  NativeProcess("locationd", "selfdrive/locationd", ["./locationd"]),
  NativeProcess("boardd", "selfdrive/boardd", ["./boardd"], enabled=False),
  PythonProcess("calibrationd", "selfdrive.locationd.calibrationd"),
  PythonProcess("torqued", "selfdrive.locationd.torqued"),
  PythonProcess("controlsd", "selfdrive.controls.controlsd"),
  PythonProcess("deleter", "system.loggerd.deleter", offroad=True),
  PythonProcess("dmonitoringd", "selfdrive.monitoring.dmonitoringd", enabled=(not PC or WEBCAM), callback=driverview),
  PythonProcess("laikad", "selfdrive.locationd.laikad"),
  PythonProcess("rawgpsd", "system.sensord.rawgps.rawgpsd", enabled=TICI),
  PythonProcess("navd", "selfdrive.navd.navd"),
  PythonProcess("pandad", "selfdrive.boardd.pandad", offroad=True),
  PythonProcess("paramsd", "selfdrive.locationd.paramsd"),
  NativeProcess("ubloxd", "system/ubloxd", ["./ubloxd"], enabled=TICI, onroad=False, callback=ublox),
  PythonProcess("pigeond", "system.sensord.pigeond", enabled=TICI, onroad=False, callback=ublox),
  PythonProcess("plannerd", "selfdrive.controls.plannerd"),
  PythonProcess("radard", "selfdrive.controls.radard"),
  PythonProcess("thermald", "selfdrive.thermald.thermald", offroad=True),
  PythonProcess("tombstoned", "selfdrive.tombstoned", enabled=not PC, offroad=True),
  PythonProcess("updated", "selfdrive.updated", enabled=not PC, onroad=False, offroad=True),
  PythonProcess("uploader", "system.loggerd.uploader", offroad=True),
  PythonProcess("statsd", "selfdrive.statsd", offroad=True),

  # debug procs
  NativeProcess("bridge", "cereal/messaging", ["./bridge"], onroad=False, callback=notcar),
  PythonProcess("webjoystick", "tools.joystick.web", onroad=False, callback=notcar),
]

managed_processes = {p.name: p for p in procs}
