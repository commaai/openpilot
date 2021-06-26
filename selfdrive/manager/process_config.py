import os

from selfdrive.manager.process import PythonProcess, NativeProcess, DaemonProcess
from selfdrive.hardware import EON, TICI, PC, JETSON
from common.params import Params

JETSON = JETSON or Params().get_bool("dp_jetson")
WEBCAM = os.getenv("USE_WEBCAM") is not None

procs = [
  DaemonProcess("manage_athenad", "selfdrive.athena.manage_athenad", "AthenadPid", enabled=not JETSON),
  # due to qualcomm kernel bugs SIGKILLing camerad sometimes causes page table corruption
  NativeProcess("camerad", "selfdrive/camerad", ["./camerad"], unkillable=True, driverview=True),
  NativeProcess("clocksd", "selfdrive/clocksd", ["./clocksd"]),
  NativeProcess("dmonitoringmodeld", "selfdrive/modeld", ["./dmonitoringmodeld"], enabled=not JETSON and (not PC or WEBCAM), driverview=True),
  NativeProcess("logcatd", "selfdrive/logcatd", ["./logcatd"], enabled=not JETSON),
  NativeProcess("loggerd", "selfdrive/loggerd", ["./loggerd"], enabled=not JETSON),
  NativeProcess("modeld", "selfdrive/modeld", ["./modeld"]),
  NativeProcess("proclogd", "selfdrive/proclogd", ["./proclogd"], enabled=not JETSON),
  NativeProcess("sensord", "selfdrive/sensord", ["./sensord"], enabled=not PC and not JETSON, persistent=EON, sigkill=EON),
  NativeProcess("ubloxd", "selfdrive/locationd", ["./ubloxd"], enabled=not JETSON and (not PC or WEBCAM)),
  NativeProcess("ui", "selfdrive/ui", ["./ui"], persistent=True, watchdog_max_dt=(10 if TICI else None)),
  NativeProcess("locationd", "selfdrive/locationd", ["./locationd"]),
  PythonProcess("calibrationd", "selfdrive.locationd.calibrationd"),
  PythonProcess("controlsd", "selfdrive.controls.controlsd"),
  PythonProcess("deleter", "selfdrive.loggerd.deleter", persistent=True),
  PythonProcess("dmonitoringd", "selfdrive.monitoring.dmonitoringd", enabled=not JETSON and (not PC or WEBCAM), driverview=True),
  PythonProcess("logmessaged", "selfdrive.logmessaged", enabled=not JETSON, persistent=True),
  PythonProcess("pandad", "selfdrive.pandad", persistent=True),
  PythonProcess("paramsd", "selfdrive.locationd.paramsd"),
  PythonProcess("plannerd", "selfdrive.controls.plannerd"),
  PythonProcess("radard", "selfdrive.controls.radard"),
  PythonProcess("rtshield", "selfdrive.rtshield", enabled=EON),
  PythonProcess("thermald", "selfdrive.thermald.thermald", persistent=True),
  PythonProcess("timezoned", "selfdrive.timezoned", enabled=TICI, persistent=True),
  PythonProcess("tombstoned", "selfdrive.tombstoned", enabled=not PC and not JETSON, persistent=True),
  PythonProcess("updated", "selfdrive.updated", enabled=not PC and not JETSON, persistent=True),
  PythonProcess("uploader", "selfdrive.loggerd.uploader", enabled=not JETSON, persistent=True),
  PythonProcess("systemd", "selfdrive.dragonpilot.systemd", persistent=True),
]

managed_processes = {p.name: p for p in procs}
