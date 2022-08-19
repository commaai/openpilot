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

procs = [
  # due to qualcomm kernel bugs SIGKILLing camerad sometimes causes page table corruption
  NativeProcess("camerad", "system/camerad", ["./camerad"], unkillable=True, callback=driverview),
  NativeProcess("clocksd", "system/clocksd", ["./clocksd"]),
  NativeProcess("logcatd", "system/logcatd", ["./logcatd"]),
  NativeProcess("proclogd", "system/proclogd", ["./proclogd"]),
  PythonProcess("logmessaged", "system.logmessaged", offroad=True),
  PythonProcess("timezoned", "system.timezoned", enabled=not PC, offroad=True),

  DaemonProcess("manage_athenad", "selfdrive.athena.manage_athenad", "AthenadPid"),
  NativeProcess("dmonitoringmodeld", "selfdrive/modeld", ["./dmonitoringmodeld"], enabled=(not PC or WEBCAM), callback=driverview),
  NativeProcess("encoderd", "selfdrive/loggerd", ["./encoderd"]),
  NativeProcess("loggerd", "selfdrive/loggerd", ["./loggerd"], onroad=False, callback=logging),
  NativeProcess("modeld", "selfdrive/modeld", ["./modeld"]),
  NativeProcess("sensord", "selfdrive/sensord", ["./sensord"], enabled=not PC),
  NativeProcess("ubloxd", "selfdrive/locationd", ["./ubloxd"], enabled=(not PC or WEBCAM)),
  NativeProcess("ui", "selfdrive/ui", ["./ui"], offroad=True, watchdog_max_dt=(5 if not PC else None)),
  NativeProcess("soundd", "selfdrive/ui/soundd", ["./soundd"], offroad=True),
  NativeProcess("locationd", "selfdrive/locationd", ["./locationd"]),
  NativeProcess("boardd", "selfdrive/boardd", ["./boardd"], enabled=False),
  PythonProcess("calibrationd", "selfdrive.locationd.calibrationd"),
  PythonProcess("controlsd", "selfdrive.controls.controlsd"),
  PythonProcess("deleter", "selfdrive.loggerd.deleter", offroad=True),
  PythonProcess("dmonitoringd", "selfdrive.monitoring.dmonitoringd", enabled=(not PC or WEBCAM), callback=driverview),
  PythonProcess("laikad", "selfdrive.locationd.laikad"),
  PythonProcess("navd", "selfdrive.navd.navd"),
  PythonProcess("pandad", "selfdrive.boardd.pandad", offroad=True),
  PythonProcess("paramsd", "selfdrive.locationd.paramsd"),
  PythonProcess("plannerd", "selfdrive.controls.plannerd"),
  PythonProcess("radard", "selfdrive.controls.radard"),
  PythonProcess("thermald", "selfdrive.thermald.thermald", offroad=True),
  PythonProcess("tombstoned", "selfdrive.tombstoned", enabled=not PC, offroad=True),
  PythonProcess("updated", "selfdrive.updated", enabled=not PC, onroad=False, offroad=True),
  PythonProcess("uploader", "selfdrive.loggerd.uploader", offroad=True),
  PythonProcess("statsd", "selfdrive.statsd", offroad=True),

  NativeProcess("bridge", "cereal/messaging", ["./bridge"], onroad=False, callback=notcar),
  PythonProcess("webjoystick", "tools.joystick.web", onroad=False, callback=notcar),

  # Experimental
  PythonProcess("rawgpsd", "selfdrive.sensord.rawgps.rawgpsd", enabled=(TICI and os.path.isfile("/persist/comma/use-quectel-rawgps"))),
]

managed_processes = {p.name: p for p in procs}
