import os

from cereal import car
from openpilot.common.params import Params
from openpilot.selfdrive.manager.process import PythonProcess, NativeProcess, DaemonProcess

WEBCAM = os.getenv("USE_WEBCAM") is not None

def driverview(started: bool, params: Params, CP: car.CarParams) -> bool:
  return started or params.get_bool("IsDriverViewEnabled")

def logging(started, params, CP: car.CarParams) -> bool:
  run = (not CP.notCar) or not params.get_bool("DisableLogging")
  return started and run

def always_run(started, params, CP: car.CarParams) -> bool:
  return True

def only_onroad(started: bool, params, CP: car.CarParams) -> bool:
  return started

def only_offroad(started, params, CP: car.CarParams) -> bool:
  return not started

ATHENA            = DaemonProcess("manage_athenad", "selfdrive.athena.manage_athenad", "AthenadPid")

LOGCAT            = NativeProcess("logcatd", "system/logcatd", ["./logcatd"], only_onroad)
PROCLOG           = NativeProcess("proclogd", "system/proclogd", ["./proclogd"], only_onroad)
LOGMESSAGED       = PythonProcess("logmessaged", "system.logmessaged", always_run)
TIMED             = PythonProcess("timed", "system.timed", always_run)
DMONITORINGMODELD = PythonProcess("dmonitoringmodeld", "selfdrive.modeld.dmonitoringmodeld", driverview)

ENCODERD          = NativeProcess("encoderd", "system/loggerd", ["./encoderd"], only_onroad)
#ENCODERD_STREAM   = NativeProcess("stream_encoderd", "system/loggerd", ["./encoderd", "--stream"], notcar)
LOGGERD           = NativeProcess("loggerd", "system/loggerd", ["./loggerd"], logging)
MODELD            = NativeProcess("modeld", "selfdrive/modeld", ["./modeld"], only_onroad)
MAPSD             = NativeProcess("mapsd", "selfdrive/navd", ["./mapsd"], only_onroad)
NAVMODELD         = PythonProcess("navmodeld", "selfdrive.modeld.navmodeld", only_onroad)
LOCATIOND         = NativeProcess("locationd", "selfdrive/locationd", ["./locationd"], only_onroad)
CALIBRATIOND      = PythonProcess("calibrationd", "selfdrive.locationd.calibrationd", only_onroad)
TORQUED           = PythonProcess("torqued", "selfdrive.locationd.torqued", only_onroad)
CONTROLSD         = PythonProcess("controlsd", "selfdrive.controls.controlsd", only_onroad)
DELETER           = PythonProcess("deleter", "system.loggerd.deleter", always_run)
DMONITORINGD      = PythonProcess("dmonitoringd", "selfdrive.monitoring.dmonitoringd", driverview)
NAVD              = PythonProcess("navd", "selfdrive.navd.navd", only_onroad)
PARAMSD           = PythonProcess("paramsd", "selfdrive.locationd.paramsd", only_onroad)

PLANNERD          = PythonProcess("plannerd", "selfdrive.controls.plannerd", only_onroad)
RADARD            = PythonProcess("radard", "selfdrive.controls.radard", only_onroad)
THERMALD          = PythonProcess("thermald", "selfdrive.thermald.thermald", always_run)
TOMBSTONED        = PythonProcess("tombstoned", "selfdrive.tombstoned", always_run)
UPDATED           = PythonProcess("updated", "selfdrive.updated", only_offroad)
UPLOADERD         = PythonProcess("uploader", "system.loggerd.uploader", always_run)
STATSD            = PythonProcess("statsd", "selfdrive.statsd", always_run)

# hardware
CAMERAD           = NativeProcess("camerad", "system/camerad", ["./camerad"], driverview)
UBLOXD            = NativeProcess("ubloxd", "system/ubloxd", ["./ubloxd"], only_onroad)
PIGEOND           = PythonProcess("pigeond", "system.sensord.pigeond", only_onroad)
QCOMGPSD          = PythonProcess("qcomgpsd", "system.qcomgpsd.qcomgpsd", only_onroad)
SENSORD           = NativeProcess("sensord", "system/sensord", ["./sensord"], only_onroad)

BOARDD            = NativeProcess("boardd", "selfdrive/boardd", ["./boardd"], always_run, enabled=False)
PANDAD            = PythonProcess("pandad", "selfdrive.boardd.pandad", always_run)

# ui
MICD              = PythonProcess("micd", "system.micd", always_run)
SOUNDD            = PythonProcess("soundd", "selfdrive.ui.soundd", only_onroad)
UI                = NativeProcess("ui", "selfdrive/ui", ["./ui"], always_run)


# debug procs
BRIDGE            = NativeProcess("bridge", "cereal/messaging", ["./bridge"], always_run)
WEBRTCD           = PythonProcess("webrtcd", "system.webrtc.webrtcd", always_run)
WEBJOYSTICKD      = PythonProcess("webjoystick", "tools.bodyteleop.web", always_run)
