from selfdrive.manager.process import PythonProcess, NativeProcess, DaemonProcess
from selfdrive.hardware import EON, TICI, PC

procs = [
  DaemonProcess("manage_athenad", "selfdrive.athena.manage_athenad", "AthenadPid"),
  # due to qualcomm kernel bugs SIGKILLing camerad sometimes causes page table corruption
  NativeProcess("camerad", "selfdrive/camerad", ["./camerad"], unkillable=True, driverview=True),
  NativeProcess("clocksd", "selfdrive/clocksd", ["./clocksd"]),
  NativeProcess("dmonitoringmodeld", "selfdrive/modeld", ["./dmonitoringmodeld"], driverview=True),
  NativeProcess("logcatd", "selfdrive/logcatd", ["./logcatd"]),
  NativeProcess("loggerd", "selfdrive/loggerd", ["./loggerd"]),
  NativeProcess("modeld", "selfdrive/modeld", ["./modeld"]),
  NativeProcess("proclogd", "selfdrive/proclogd", ["./proclogd"]),
  NativeProcess("ubloxd", "selfdrive/locationd", ["./ubloxd"]),
  NativeProcess("ui", "selfdrive/ui", ["./ui"], persistent=True),
  PythonProcess("calibrationd", "selfdrive.locationd.calibrationd"),
  PythonProcess("controlsd", "selfdrive.controls.controlsd"),
  PythonProcess("deleter", "selfdrive.loggerd.deleter", persistent=True),
  PythonProcess("dmonitoringd", "selfdrive.monitoring.dmonitoringd", driverview=True),
  PythonProcess("locationd", "selfdrive.locationd.locationd"),
  PythonProcess("logmessaged", "selfdrive.logmessaged", persistent=True),
  PythonProcess("pandad", "selfdrive.pandad", persistent=True),
  PythonProcess("paramsd", "selfdrive.locationd.paramsd"),
  PythonProcess("plannerd", "selfdrive.controls.plannerd"),
  PythonProcess("radard", "selfdrive.controls.radard"),
  PythonProcess("thermald", "selfdrive.thermald.thermald", persistent=True),
  PythonProcess("uploader", "selfdrive.loggerd.uploader", persistent=True),
]

if not PC:
  procs += [
    NativeProcess("sensord", "selfdrive/sensord", ["./sensord"], persistent=EON, sigkill=EON),
    PythonProcess("tombstoned", "selfdrive.tombstoned", persistent=True),
    PythonProcess("updated", "selfdrive.updated", persistent=True),
  ]

if TICI:
  procs += [
    PythonProcess("timezoned", "selfdrive.timezoned", persistent=True),
  ]

if EON:
  procs += [
    PythonProcess("rtshield", "selfdrive.rtshield"),
  ]


managed_processes = {p.name: p for p in procs}
