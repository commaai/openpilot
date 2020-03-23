#!/usr/bin/env python3.7
import os
import time
import sys
import fcntl
import errno
import signal
import shutil
import subprocess
import datetime

from common.basedir import BASEDIR, PARAMS
from common.android import ANDROID
sys.path.append(os.path.join(BASEDIR, "pyextra"))
os.environ['BASEDIR'] = BASEDIR

TOTAL_SCONS_NODES = 1195
prebuilt = os.path.exists(os.path.join(BASEDIR, 'prebuilt'))

# Create folders needed for msgq
try:
  os.mkdir("/dev/shm")
except FileExistsError:
  pass
except PermissionError:
  print("WARNING: failed to make /dev/shm")

if ANDROID:
  os.chmod("/dev/shm", 0o777)

def unblock_stdout():
  # get a non-blocking stdout
  child_pid, child_pty = os.forkpty()
  if child_pid != 0: # parent

    # child is in its own process group, manually pass kill signals
    signal.signal(signal.SIGINT, lambda signum, frame: os.kill(child_pid, signal.SIGINT))
    signal.signal(signal.SIGTERM, lambda signum, frame: os.kill(child_pid, signal.SIGTERM))

    fcntl.fcntl(sys.stdout, fcntl.F_SETFL,
       fcntl.fcntl(sys.stdout, fcntl.F_GETFL) | os.O_NONBLOCK)

    while True:
      try:
        dat = os.read(child_pty, 4096)
      except OSError as e:
        if e.errno == errno.EIO:
          break
        continue

      if not dat:
        break

      try:
        sys.stdout.write(dat.decode('utf8'))
      except (OSError, IOError, UnicodeDecodeError):
        pass

    # os.wait() returns a tuple with the pid and a 16 bit value
    # whose low byte is the signal number and whose high byte is the exit satus
    exit_status = os.wait()[1] >> 8
    os._exit(exit_status)


if __name__ == "__main__":
  unblock_stdout()
  from common.spinner import Spinner
else:
  from common.spinner import FakeSpinner as Spinner

import importlib
import traceback
from multiprocessing import Process

# Run scons
spinner = Spinner()
spinner.update("0")

if not prebuilt:
  for retry in [True, False]:
    # run scons
    env = os.environ.copy()
    env['SCONS_PROGRESS'] = "1"
    env['SCONS_CACHE'] = "1"

    nproc = os.cpu_count()
    j_flag = "" if nproc is None else "-j%d" % (nproc - 1)
    scons = subprocess.Popen(["scons", j_flag], cwd=BASEDIR, env=env, stderr=subprocess.PIPE)

    # Read progress from stderr and update spinner
    while scons.poll() is None:
      try:
        line = scons.stderr.readline()
        if line is None:
          continue

        line = line.rstrip()
        prefix = b'progress: '
        if line.startswith(prefix):
          i = int(line[len(prefix):])
          if spinner is not None:
            spinner.update("%d" % (50.0 * (i / TOTAL_SCONS_NODES)))
        elif len(line):
          print(line.decode('utf8'))
      except Exception:
        pass

    if scons.returncode != 0:
      if retry:
        print("scons build failed, cleaning in")
        for i in range(3,-1,-1):
          print("....%d" % i)
          time.sleep(1)
        subprocess.check_call(["scons", "-c"], cwd=BASEDIR, env=env)
        shutil.rmtree("/tmp/scons_cache")
      else:
        raise RuntimeError("scons build failed")
    else:
      break

import cereal
import cereal.messaging as messaging

from common.params import Params
import selfdrive.crash as crash
from selfdrive.swaglog import cloudlog
from selfdrive.registration import register
from selfdrive.version import version, dirty
from selfdrive.loggerd.config import ROOT
from selfdrive.launcher import launcher
from common import android
from common.apk import update_apks, pm_apply_packages, start_offroad
from common.manager_helpers import print_cpu_usage

ThermalStatus = cereal.log.ThermalData.ThermalStatus

# comment out anything you don't want to run
managed_processes = {
  "thermald": "selfdrive.thermald.thermald",
  "uploader": "selfdrive.loggerd.uploader",
  "deleter": "selfdrive.loggerd.deleter",
  "controlsd": "selfdrive.controls.controlsd",
  "plannerd": "selfdrive.controls.plannerd",
  "radard": "selfdrive.controls.radard",
  "dmonitoringd": "selfdrive.controls.dmonitoringd",
  "ubloxd": ("selfdrive/locationd", ["./ubloxd"]),
  "loggerd": ("selfdrive/loggerd", ["./loggerd"]),
  "logmessaged": "selfdrive.logmessaged",
  "locationd": "selfdrive.locationd.locationd",
  "tombstoned": "selfdrive.tombstoned",
  "logcatd": ("selfdrive/logcatd", ["./logcatd"]),
  "proclogd": ("selfdrive/proclogd", ["./proclogd"]),
  "boardd": ("selfdrive/boardd", ["./boardd"]),   # not used directly
  "pandad": "selfdrive.pandad",
  "ui": ("selfdrive/ui", ["./ui"]),
  "calibrationd": "selfdrive.locationd.calibrationd",
  "paramsd": ("selfdrive/locationd", ["./paramsd"]),
  "camerad": ("selfdrive/camerad", ["./camerad"]),
  "sensord": ("selfdrive/sensord", ["./sensord"]),
  "clocksd": ("selfdrive/clocksd", ["./clocksd"]),
  "gpsd": ("selfdrive/sensord", ["./gpsd"]),
  "updated": "selfdrive.updated",
  "dmonitoringmodeld": ("selfdrive/modeld", ["./dmonitoringmodeld"]),
  "modeld": ("selfdrive/modeld", ["./modeld"]),
}

daemon_processes = {
  "manage_athenad": ("selfdrive.athena.manage_athenad", "AthenadPid"),
}

running = {}
def get_running():
  return running

# due to qualcomm kernel bugs SIGKILLing camerad sometimes causes page table corruption
unkillable_processes = ['camerad']

# processes to end with SIGINT instead of SIGTERM
interrupt_processes = []

# processes to end with SIGKILL instead of SIGTERM
kill_processes = ['sensord', 'paramsd']

# processes to end if thermal conditions exceed Green parameters
green_temp_processes = ['uploader']

persistent_processes = [
  'thermald',
  'logmessaged',
  'ui',
  'uploader',
]
if ANDROID:
  persistent_processes += [
    'logcatd',
    'tombstoned',
    'updated',
  ]

car_started_processes = [
  'controlsd',
  'plannerd',
  'loggerd',
  'radard',
  'dmonitoringd',
  'calibrationd',
  'paramsd',
  'camerad',
  'modeld',
  'proclogd',
  'ubloxd',
  'locationd',
]
if ANDROID:
  car_started_processes += [
    'sensord',
    'clocksd',
    'gpsd',
    'dmonitoringmodeld',
    'deleter',
  ]

def register_managed_process(name, desc, car_started=False):
  global managed_processes, car_started_processes, persistent_processes
  print("registering %s" % name)
  managed_processes[name] = desc
  if car_started:
    car_started_processes.append(name)
  else:
    persistent_processes.append(name)

# ****************** process management functions ******************
def nativelauncher(pargs, cwd):
  # exec the process
  os.chdir(cwd)

  # because when extracted from pex zips permissions get lost -_-
  os.chmod(pargs[0], 0o700)

  os.execvp(pargs[0], pargs)

def start_managed_process(name):
  if name in running or name not in managed_processes:
    return
  proc = managed_processes[name]
  if isinstance(proc, str):
    cloudlog.info("starting python %s" % proc)
    running[name] = Process(name=name, target=launcher, args=(proc,))
  else:
    pdir, pargs = proc
    cwd = os.path.join(BASEDIR, pdir)
    cloudlog.info("starting process %s" % name)
    running[name] = Process(name=name, target=nativelauncher, args=(pargs, cwd))
  running[name].start()

def start_daemon_process(name):
  params = Params()
  proc, pid_param = daemon_processes[name]
  pid = params.get(pid_param, encoding='utf-8')

  if pid is not None:
    try:
      os.kill(int(pid), 0)
      with open(f'/proc/{pid}/cmdline') as f:
        if proc in f.read():
          # daemon is running
          return
    except (OSError, FileNotFoundError):
      # process is dead
      pass

  cloudlog.info("starting daemon %s" % name)
  proc = subprocess.Popen(['python', '-m', proc],
                         stdin=open('/dev/null', 'r'),
                         stdout=open('/dev/null', 'w'),
                         stderr=open('/dev/null', 'w'),
                         preexec_fn=os.setpgrp)

  params.put(pid_param, str(proc.pid))

def prepare_managed_process(p):
  proc = managed_processes[p]
  if isinstance(proc, str):
    # import this python
    cloudlog.info("preimporting %s" % proc)
    importlib.import_module(proc)
  elif os.path.isfile(os.path.join(BASEDIR, proc[0], "Makefile")):
    # build this process
    cloudlog.info("building %s" % (proc,))
    try:
      subprocess.check_call(["make", "-j4"], cwd=os.path.join(BASEDIR, proc[0]))
    except subprocess.CalledProcessError:
      # make clean if the build failed
      cloudlog.warning("building %s failed, make clean" % (proc, ))
      subprocess.check_call(["make", "clean"], cwd=os.path.join(BASEDIR, proc[0]))
      subprocess.check_call(["make", "-j4"], cwd=os.path.join(BASEDIR, proc[0]))


def join_process(process, timeout):
  # Process().join(timeout) will hang due to a python 3 bug: https://bugs.python.org/issue28382
  # We have to poll the exitcode instead
  t = time.time()
  while time.time() - t < timeout and process.exitcode is None:
    time.sleep(0.001)


def kill_managed_process(name):
  if name not in running or name not in managed_processes:
    return
  cloudlog.info("killing %s" % name)

  if running[name].exitcode is None:
    if name in interrupt_processes:
      os.kill(running[name].pid, signal.SIGINT)
    elif name in kill_processes:
      os.kill(running[name].pid, signal.SIGKILL)
    else:
      running[name].terminate()

    join_process(running[name], 5)

    if running[name].exitcode is None:
      if name in unkillable_processes:
        cloudlog.critical("unkillable process %s failed to exit! rebooting in 15 if it doesn't die" % name)
        join_process(running[name], 15)
        if running[name].exitcode is None:
          cloudlog.critical("FORCE REBOOTING PHONE!")
          os.system("date >> /sdcard/unkillable_reboot")
          os.system("reboot")
          raise RuntimeError
      else:
        cloudlog.info("killing %s with SIGKILL" % name)
        os.kill(running[name].pid, signal.SIGKILL)
        running[name].join()

  cloudlog.info("%s is dead with %d" % (name, running[name].exitcode))
  del running[name]


def cleanup_all_processes(signal, frame):
  cloudlog.info("caught ctrl-c %s %s" % (signal, frame))

  if ANDROID:
    pm_apply_packages('disable')

  for name in list(running.keys()):
    kill_managed_process(name)
  cloudlog.info("everything is dead")

# ****************** run loop ******************

def manager_init(should_register=True):
  if should_register:
    reg_res = register()
    if reg_res:
      dongle_id = reg_res
    else:
      raise Exception("server registration failed")
  else:
    dongle_id = "c"*16

  # set dongle id
  cloudlog.info("dongle id is " + dongle_id)
  os.environ['DONGLE_ID'] = dongle_id

  cloudlog.info("dirty is %d" % dirty)
  if not dirty:
    os.environ['CLEAN'] = '1'

  cloudlog.bind_global(dongle_id=dongle_id, version=version, dirty=dirty, is_eon=True)
  crash.bind_user(id=dongle_id)
  crash.bind_extra(version=version, dirty=dirty, is_eon=True)

  os.umask(0)
  try:
    os.mkdir(ROOT, 0o777)
  except OSError:
    pass

  # ensure shared libraries are readable by apks
  if ANDROID:
    os.chmod(BASEDIR, 0o755)
    os.chmod(os.path.join(BASEDIR, "cereal"), 0o755)
    os.chmod(os.path.join(BASEDIR, "cereal", "libmessaging_shared.so"), 0o755)

def manager_thread():
  # now loop
  thermal_sock = messaging.sub_sock('thermal')

  if os.getenv("GET_CPU_USAGE"):
    proc_sock = messaging.sub_sock('procLog', conflate=True)

  cloudlog.info("manager start")
  cloudlog.info({"environ": os.environ})

  # save boot log
  subprocess.call(["./loggerd", "--bootlog"], cwd=os.path.join(BASEDIR, "selfdrive/loggerd"))

  params = Params()

  # start daemon processes
  for p in daemon_processes:
    start_daemon_process(p)

  # start persistent processes
  for p in persistent_processes:
    start_managed_process(p)

  # start offroad
  if ANDROID:
    pm_apply_packages('enable')
    start_offroad()

  if os.getenv("NOBOARD") is None:
    start_managed_process("pandad")

  if os.getenv("BLOCK") is not None:
    for k in os.getenv("BLOCK").split(","):
      del managed_processes[k]

  logger_dead = False

  start_t = time.time()
  first_proc = None

  while 1:
    msg = messaging.recv_sock(thermal_sock, wait=True)

    # heavyweight batch processes are gated on favorable thermal conditions
    if msg.thermal.thermalStatus >= ThermalStatus.yellow:
      for p in green_temp_processes:
        if p in persistent_processes:
          kill_managed_process(p)
    else:
      for p in green_temp_processes:
        if p in persistent_processes:
          start_managed_process(p)

    if msg.thermal.freeSpace < 0.05:
      logger_dead = True

    if msg.thermal.started:
      for p in car_started_processes:
        if p == "loggerd" and logger_dead:
          kill_managed_process(p)
        else:
          start_managed_process(p)
    else:
      logger_dead = False
      for p in reversed(car_started_processes):
        kill_managed_process(p)

    # check the status of all processes, did any of them die?
    running_list = ["%s%s\u001b[0m" % ("\u001b[32m" if running[p].is_alive() else "\u001b[31m", p) for p in running]
    cloudlog.debug(' '.join(running_list))

    # Exit main loop when uninstall is needed
    if params.get("DoUninstall", encoding='utf8') == "1":
      break

    if os.getenv("GET_CPU_USAGE"):
      dt = time.time() - start_t

      # Get first sample
      if dt > 30 and first_proc is None:
        first_proc = messaging.recv_sock(proc_sock)

      # Get last sample and exit
      if dt > 90:
        first_proc = first_proc
        last_proc = messaging.recv_sock(proc_sock, wait=True)

        cleanup_all_processes(None, None)
        sys.exit(print_cpu_usage(first_proc, last_proc))

def manager_prepare(spinner=None):
  # build all processes
  os.chdir(os.path.dirname(os.path.abspath(__file__)))

  # Spinner has to start from 70 here
  total = 100.0 if prebuilt else 50.0

  for i, p in enumerate(managed_processes):
    if spinner is not None:
      spinner.update("%d" % ((100.0 - total) + total * (i + 1) / len(managed_processes),))
    prepare_managed_process(p)

def uninstall():
  cloudlog.warning("uninstalling")
  with open('/cache/recovery/command', 'w') as f:
    f.write('--wipe_data\n')
  # IPowerManager.reboot(confirm=false, reason="recovery", wait=true)
  android.reboot(reason="recovery")

def main():
  os.environ['PARAMS_PATH'] = PARAMS

  # the flippening!
  os.system('LD_LIBRARY_PATH="" content insert --uri content://settings/system --bind name:s:user_rotation --bind value:i:1')

  # disable bluetooth
  os.system('service call bluetooth_manager 8')

  params = Params()
  params.manager_start()

  default_params = [
    ("CommunityFeaturesToggle", "0"),
    ("CompletedTrainingVersion", "0"),
    ("IsMetric", "0"),
    ("RecordFront", "0"),
    ("HasAcceptedTerms", "0"),
    ("HasCompletedSetup", "0"),
    ("IsUploadRawEnabled", "1"),
    ("IsLdwEnabled", "1"),
    ("IsGeofenceEnabled", "-1"),
    ("SpeedLimitOffset", "0"),
    ("LongitudinalControl", "0"),
    ("LimitSetSpeed", "0"),
    ("LimitSetSpeedNeural", "0"),
    ("LastUpdateTime", datetime.datetime.utcnow().isoformat().encode('utf8')),
    ("OpenpilotEnabledToggle", "1"),
    ("LaneChangeEnabled", "1"),
  ]

  # set unset params
  for k, v in default_params:
    if params.get(k) is None:
      params.put(k, v)

  # is this chffrplus?
  if os.getenv("PASSIVE") is not None:
    params.put("Passive", str(int(os.getenv("PASSIVE"))))

  if params.get("Passive") is None:
    raise Exception("Passive must be set to continue")

  if ANDROID:
    update_apks()
  manager_init()
  manager_prepare(spinner)
  spinner.close()

  if os.getenv("PREPAREONLY") is not None:
    return

  # SystemExit on sigterm
  signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(1))

  try:
    manager_thread()
  except SystemExit:
    raise
  except Exception:
    traceback.print_exc()
    crash.capture_exception()
  finally:
    cleanup_all_processes(None, None)

  if params.get("DoUninstall", encoding='utf8') == "1":
    uninstall()

if __name__ == "__main__":
  main()
  # manual exit because we are forked
  sys.exit(0)
