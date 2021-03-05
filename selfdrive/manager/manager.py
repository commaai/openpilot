#!/usr/bin/env python3
import datetime
import os
import signal
import subprocess
import sys
import traceback

import cereal.messaging as messaging
import selfdrive.crash as crash
from common.basedir import BASEDIR
from common.params import Params
from common.spinner import Spinner
from common.text_window import TextWindow
from selfdrive.hardware import EON, HARDWARE
from selfdrive.hardware.eon.apk import (pm_apply_packages, start_offroad,
                                        update_apks)
from selfdrive.manager.build import MAX_BUILD_PROGRESS, PREBUILT
from selfdrive.manager.helpers import unblock_stdout
from selfdrive.manager.process import ensure_running
from selfdrive.manager.process_config import managed_processes
from selfdrive.registration import register
from selfdrive.swaglog import add_logentries_handler, cloudlog
from selfdrive.version import dirty, version


def manager_init(spinner=None):
  params = Params()
  params.manager_start()

  default_params = [
    ("CommunityFeaturesToggle", "0"),
    ("CompletedTrainingVersion", "0"),
    ("IsRHD", "0"),
    ("IsMetric", "0"),
    ("RecordFront", "0"),
    ("HasAcceptedTerms", "0"),
    ("HasCompletedSetup", "0"),
    ("IsUploadRawEnabled", "1"),
    ("IsLdwEnabled", "1"),
    ("LastUpdateTime", datetime.datetime.utcnow().isoformat().encode('utf8')),
    ("OpenpilotEnabledToggle", "1"),
    ("VisionRadarToggle", "0"),
    ("LaneChangeEnabled", "1"),
    ("IsDriverViewEnabled", "0"),
  ]

  # set unset params
  for k, v in default_params:
    if params.get(k) is None:
      params.put(k, v)

  # is this dashcam?
  if os.getenv("PASSIVE") is not None:
    params.put("Passive", str(int(os.getenv("PASSIVE"))))

  if params.get("Passive") is None:
    raise Exception("Passive must be set to continue")

  if EON:
    update_apks()

  os.umask(0)  # Make sure we can create files with 777 permissions

  # Create folders needed for msgq
  try:
    os.mkdir("/dev/shm")
  except FileExistsError:
    pass
  except PermissionError:
    print("WARNING: failed to make /dev/shm")

  # set dongle id
  reg_res = register(spinner)
  if reg_res:
    dongle_id = reg_res
  else:
    raise Exception("server registration failed")
  os.environ['DONGLE_ID'] = dongle_id  # Needed for swaglog and loggerd

  if not dirty:
    os.environ['CLEAN'] = '1'

  cloudlog.bind_global(dongle_id=dongle_id, version=version, dirty=dirty,
                       device=HARDWARE.get_device_type())
  crash.bind_user(id=dongle_id)
  crash.bind_extra(version=version, dirty=dirty, device=HARDWARE.get_device_type())

  # ensure shared libraries are readable by apks
  if EON:
    os.chmod(BASEDIR, 0o755)
    os.chmod("/dev/shm", 0o777)
    os.chmod(os.path.join(BASEDIR, "cereal"), 0o755)
    os.chmod(os.path.join(BASEDIR, "cereal", "libmessaging_shared.so"), 0o755)


def manager_prepare(spinner=None):
  # build all processes
  os.chdir(os.path.dirname(os.path.abspath(__file__)))

  total = 100.0 - (0 if PREBUILT else MAX_BUILD_PROGRESS)

  for i, p in enumerate(managed_processes.values()):
    perc = (100.0 - total) + total * (i + 1) / len(managed_processes)

    if spinner:
      spinner.update_progress(perc, 100.)
    p.prepare()


def manager_cleanup():
  if EON:
    pm_apply_packages('disable')

  for p in managed_processes.values():
    p.stop()

  cloudlog.info("everything is dead")


def manager_thread():
  cloudlog.info("manager start")
  cloudlog.info({"environ": os.environ})

  # save boot log
  subprocess.call("./bootlog", cwd=os.path.join(BASEDIR, "selfdrive/loggerd"))

  ignore = []
  if os.getenv("NOBOARD") is not None:
    ignore.append("pandad")
  if os.getenv("BLOCK") is not None:
    ignore += os.getenv("BLOCK").split(",")

  # start offroad
  if EON and "QT" not in os.environ:
    pm_apply_packages('enable')
    start_offroad()

  started_prev = False
  params = Params()
  sm = messaging.SubMaster(['deviceState'])
  pm = messaging.PubMaster(['managerState'])

  while True:
    sm.update()
    not_run = ignore[:]

    if sm['deviceState'].freeSpacePercent < 5:
      not_run.append("loggerd")

    started = sm['deviceState'].started
    driverview = params.get("IsDriverViewEnabled") == b"1"
    ensure_running(managed_processes.values(), started, driverview, not_run)

    # trigger an update after going offroad
    if started_prev and not started:
      os.sync()
      managed_processes['updated'].signal(signal.SIGHUP)

    started_prev = started

    running_list = ["%s%s\u001b[0m" % ("\u001b[32m" if p.proc.is_alive() else "\u001b[31m", p.name)
                    for p in managed_processes.values() if p.proc]
    cloudlog.debug(' '.join(running_list))

    # send managerState
    msg = messaging.new_message('managerState')
    msg.managerState.processes = [p.get_process_state_msg() for p in managed_processes.values()]
    pm.send('managerState', msg)

    # Exit main loop when uninstall is needed
    if params.get("DoUninstall", encoding='utf8') == "1":
      break


def main(spinner=None):
  manager_init(spinner)
  manager_prepare(spinner)

  if spinner:
    spinner.close()

  if os.getenv("PREPAREONLY") is not None:
    return

  # SystemExit on sigterm
  signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(1))

  try:
    manager_thread()
  except Exception:
    traceback.print_exc()
    crash.capture_exception()
  finally:
    manager_cleanup()

  if Params().params.get("DoUninstall", encoding='utf8') == "1":
    cloudlog.warning("uninstalling")
    HARDWARE.uninstall()


if __name__ == "__main__":
  unblock_stdout()
  spinner = Spinner()

  try:
    main(spinner)
  except Exception:
    add_logentries_handler(cloudlog)
    cloudlog.exception("Manager failed to start")

    # Show last 3 lines of traceback
    error = traceback.format_exc(-3)
    error = "Manager failed to start\n\n" + error
    spinner.close()
    with TextWindow(error) as t:
      t.wait_for_exit()

    raise

  # manual exit because we are forked
  sys.exit(0)
