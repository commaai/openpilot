#!/usr/bin/env python3
# import time
# import cereal.messaging as messaging
# from selfdrive.manager import start_managed_process, start_daemon_process, kill_managed_process
from selfdrive.manager import start_managed_process

# services = ['controlsState', 'thermal', 'radarState']  # the services needed to be spoofed to start ui offroad
managed_processes = {
#   "thermald": "selfdrive.thermald.thermald",
#   "uploader": "selfdrive.loggerd.uploader",
#   "deleter": "selfdrive.loggerd.deleter",
  "controlsd": "selfdrive.controls.controlsd",
  "plannerd": "selfdrive.controls.plannerd",
  "radard": "selfdrive.controls.radard",
  "ui": ("selfdrive/ui", ["./ui"]),
  "calibrationd": "selfdrive.locationd.calibrationd",
  "camerad": ("selfdrive/camerad", ["./camerad"]),
  "modeld": ("selfdrive/modeld", ["./modeld"]),
  "speedlimitd": "selfdrive.locationd.speedlimitd",
#   "locationd": "selfdrive.locationd.locationd",
#   "dmonitoringd": "selfdrive.monitoring.dmonitoringd",
#   "ubloxd": ("selfdrive/locationd", ["./ubloxd"]),
#   "loggerd": ("selfdrive/loggerd", ["./loggerd"]),
#   "logmessaged": "selfdrive.logmessaged",
#   "tombstoned": "selfdrive.tombstoned",
#   "logcatd": ("selfdrive/logcatd", ["./logcatd"]),
#   "proclogd": ("selfdrive/proclogd", ["./proclogd"]),
#   "pandad": "selfdrive.pandad",
#   "paramsd": "selfdrive.locationd.paramsd",
#   "sensord": ("selfdrive/sensord", ["./sensord"]),
#   "clocksd": ("selfdrive/clocksd", ["./clocksd"]),
#   "gpsd": ("selfdrive/sensord", ["./gpsd"]),
#   "updated": "selfdrive.updated",
#   "dmonitoringmodeld": ("selfdrive/modeld", ["./dmonitoringmodeld"]),
#   "rtshield": "selfdrive.rtshield",
}

# start persistent processes
for p in managed_processes:
    start_managed_process(p)
# procs = ['camerad', 'ui', 'modeld', 'calibrationd', 'plannerd', 'controlsd', 'radard', 'speedlimitd']
# [start_managed_process(p) for p in procs]  # start needed processes
# pm = messaging.PubMaster(services)

# dat_cs, dat_thermal, dat_radar = [messaging.new_message(s) for s in services]
# dat_cs.controlsState.rearViewCam = False  # ui checks for these two messages
# dat_thermal.thermal.started = True

# try:
#   while True:
#     pm.send('controlsState', dat_cs)
#     # pm.send('thermal', dat_thermal)
#     pm.send('radarState', dat_radar)
#     time.sleep(1 / 100)  # continually send, rate doesn't matter for thermal
# except KeyboardInterrupt:
#   [kill_managed_process(p) for p in procs]
