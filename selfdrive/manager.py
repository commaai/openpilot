#!/usr/bin/env python3.7
import os
import time
import sys
import fcntl
import errno
import signal
import subprocess
import datetime
from common.spinner import Spinner

from common.basedir import BASEDIR
sys.path.append(os.path.join(BASEDIR, "pyextra"))
os.environ['BASEDIR'] = BASEDIR

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
      except (OSError, IOError):
        pass

    os._exit(os.wait()[1])

if __name__ == "__main__":
  unblock_stdout()

import glob
import shutil
import hashlib
import importlib
import traceback
from multiprocessing import Process

from setproctitle import setproctitle  #pylint: disable=no-name-in-module

from common.params import Params
import cereal
ThermalStatus = cereal.log.ThermalData.ThermalStatus

from selfdrive.swaglog import cloudlog
import selfdrive.messaging as messaging
from selfdrive.registration import register
from selfdrive.version import version, dirty
import selfdrive.crash as crash

from selfdrive.loggerd.config import ROOT

# comment out anything you don't want to run
managed_processes = {
  "thermald": "selfdrive.thermald",
  "uploader": "selfdrive.loggerd.uploader",
  "deleter": "selfdrive.loggerd.deleter",
  "controlsd": "selfdrive.controls.controlsd",
  "plannerd": "selfdrive.controls.plannerd",
  "radard": "selfdrive.controls.radard",
  "ubloxd": ("selfdrive/locationd", ["./ubloxd"]),
  "loggerd": ("selfdrive/loggerd", ["./loggerd"]),
  "logmessaged": "selfdrive.logmessaged",
  "tombstoned": "selfdrive.tombstoned",
  "logcatd": ("selfdrive/logcatd", ["./logcatd"]),
  "proclogd": ("selfdrive/proclogd", ["./proclogd"]),
  "boardd": ("selfdrive/boardd", ["./boardd"]),   # not used directly
  "pandad": "selfdrive.pandad",
  "ui": ("selfdrive/ui", ["./start.py"]),
  "calibrationd": "selfdrive.locationd.calibrationd",
  "paramsd": ("selfdrive/locationd", ["./paramsd"]),
  "visiond": ("selfdrive/visiond", ["./start.py"]),
  "sensord": ("selfdrive/sensord", ["./start_sensord.py"]),
  "gpsd": ("selfdrive/sensord", ["./start_gpsd.py"]),
  "updated": "selfdrive.updated",
}
daemon_processes = {
  "manage_athenad": ("selfdrive.athena.manage_athenad", "AthenadPid"),
}
android_packages = ("ai.comma.plus.offroad", "ai.comma.plus.frame")

running = {}
def get_running():
  return running

# due to qualcomm kernel bugs SIGKILLing visiond sometimes causes page table corruption
unkillable_processes = ['visiond']

# processes to end with SIGINT instead of SIGTERM
interrupt_processes = []

# processes to end with SIGKILL instead of SIGTERM
kill_processes = ['sensord', 'paramsd']

persistent_processes = [
  'thermald',
  'logmessaged',
  'logcatd',
  'tombstoned',
  'uploader',
  'ui',
  'updated',
]

car_started_processes = [
  'controlsd',
  'plannerd',
  'loggerd',
  'sensord',
  'radard',
  'calibrationd',
  'paramsd',
  'visiond',
  'proclogd',
  'ubloxd',
  'gpsd',
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
def launcher(proc):
  try:
    # import the process
    mod = importlib.import_module(proc)

    # rename the process
    setproctitle(proc)

    # create now context since we forked
    messaging.context = messaging.Context()

    # exec the process
    mod.main()
  except KeyboardInterrupt:
    cloudlog.warning("child %s got SIGINT" % proc)
  except Exception:
    # can't install the crash handler becuase sys.excepthook doesn't play nice
    # with threads, so catch it here.
    crash.capture_exception()
    raise

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

def start_daemon_process(name, params):
  proc, pid_param = daemon_processes[name]
  pid = params.get(pid_param)

  if pid is not None:
    try:
      os.kill(int(pid), 0)
      # process is running (kill is a poorly-named system call)
      return
    except OSError:
      # process is dead
      pass

  cloudlog.info("starting daemon %s" % name)
  proc = subprocess.Popen(['python', '-m', proc],
                         cwd='/',
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
  else:
    # build this process
    cloudlog.info("building %s" % (proc,))
    try:
      subprocess.check_call(["make", "-j4"], cwd=os.path.join(BASEDIR, proc[0]))
    except subprocess.CalledProcessError:
      # make clean if the build failed
      cloudlog.warning("building %s failed, make clean" % (proc, ))
      subprocess.check_call(["make", "clean"], cwd=os.path.join(BASEDIR, proc[0]))
      subprocess.check_call(["make", "-j4"], cwd=os.path.join(BASEDIR, proc[0]))

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

    # Process().join(timeout) will hang due to a python 3 bug: https://bugs.python.org/issue28382
    # We have to poll the exitcode instead
    # running[name].join(5.0)

    t = time.time()
    while time.time() - t < 5 and running[name].exitcode is None:
      time.sleep(0.001)

    if running[name].exitcode is None:
      if name in unkillable_processes:
        cloudlog.critical("unkillable process %s failed to exit! rebooting in 15 if it doesn't die" % name)
        running[name].join(15.0)
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

def pm_apply_packages(cmd):
  for p in android_packages:
    system("pm %s %s" % (cmd, p))

def cleanup_all_processes(signal, frame):
  cloudlog.info("caught ctrl-c %s %s" % (signal, frame))

  pm_apply_packages('disable')

  for name in list(running.keys()):
    kill_managed_process(name)
  cloudlog.info("everything is dead")


# ****************** run loop ******************

def manager_init(should_register=True):
  if should_register:
    reg_res = register()
    if reg_res:
      dongle_id, dongle_secret = reg_res
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

def system(cmd):
  try:
    cloudlog.info("running %s" % cmd)
    subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
  except subprocess.CalledProcessError as e:
    cloudlog.event("running failed",
      cmd=e.cmd,
      output=e.output[-1024:],
      returncode=e.returncode)


def manager_thread():
  # now loop
  thermal_sock = messaging.sub_sock('thermal')

  cloudlog.info("manager start")
  cloudlog.info({"environ": os.environ})

  # save boot log
  subprocess.call(["./loggerd", "--bootlog"], cwd=os.path.join(BASEDIR, "selfdrive/loggerd"))

  params = Params()

  # start daemon processes
  for p in daemon_processes:
    start_daemon_process(p, params)

  # start persistent processes
  for p in persistent_processes:
    start_managed_process(p)

  # start frame
  pm_apply_packages('enable')
  system("LD_LIBRARY_PATH= appops set ai.comma.plus.offroad SU allow")
  system("am start -n ai.comma.plus.frame/.MainActivity")

  if os.getenv("NOBOARD") is None:
    start_managed_process("pandad")

  logger_dead = False

  while 1:
    msg = messaging.recv_sock(thermal_sock, wait=True)

    # uploader is gated based on the phone temperature
    if msg.thermal.thermalStatus >= ThermalStatus.yellow:
      kill_managed_process("uploader")
    else:
      start_managed_process("uploader")

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
      for p in car_started_processes:
        kill_managed_process(p)

    # check the status of all processes, did any of them die?
    running_list = ["   running %s %s" % (p, running[p]) for p in running]
    cloudlog.debug('\n'.join(running_list))

    # Exit main loop when uninstall is needed
    if params.get("DoUninstall", encoding='utf8') == "1":
      break

def get_installed_apks():
  dat = subprocess.check_output(["pm", "list", "packages", "-f"], encoding='utf8').strip().split("\n")  # pylint: disable=unexpected-keyword-arg
  ret = {}
  for x in dat:
    if x.startswith("package:"):
      v,k = x.split("package:")[1].split("=")
      ret[k] = v
  return ret

def install_apk(path):
  # can only install from world readable path
  install_path = "/sdcard/%s" % os.path.basename(path)
  shutil.copyfile(path, install_path)

  ret = subprocess.call(["pm", "install", "-r", install_path])
  os.remove(install_path)
  return ret == 0

def update_apks():
  # install apks
  installed = get_installed_apks()

  install_apks = glob.glob(os.path.join(BASEDIR, "apk/*.apk"))
  for apk in install_apks:
    app = os.path.basename(apk)[:-4]
    if app not in installed:
      installed[app] = None

  cloudlog.info("installed apks %s" % (str(installed), ))

  for app in installed.keys():

    apk_path = os.path.join(BASEDIR, "apk/"+app+".apk")
    if not os.path.exists(apk_path):
      continue

    h1 = hashlib.sha1(open(apk_path, 'rb').read()).hexdigest()
    h2 = None
    if installed[app] is not None:
      h2 = hashlib.sha1(open(installed[app], 'rb').read()).hexdigest()
      cloudlog.info("comparing version of %s  %s vs %s" % (app, h1, h2))

    if h2 is None or h1 != h2:
      cloudlog.info("installing %s" % app)

      success = install_apk(apk_path)
      if not success:
        cloudlog.info("needing to uninstall %s" % app)
        system("pm uninstall %s" % app)
        success = install_apk(apk_path)

      assert success

def manager_update():
  update_apks()

  uninstall = [app for app in get_installed_apks().keys() if app in ("com.spotify.music", "com.waze")]
  for app in uninstall:
    cloudlog.info("uninstalling %s" % app)
    os.system("pm uninstall % s" % app)

def manager_prepare(spinner=None):
  # build cereal first
  subprocess.check_call(["make", "-j4"], cwd=os.path.join(BASEDIR, "cereal"))

  # build all processes
  os.chdir(os.path.dirname(os.path.abspath(__file__)))

  for i, p in enumerate(managed_processes):
    if spinner is not None:
      spinner.update("%d" % (100.0 * (i + 1) / len(managed_processes),))
    prepare_managed_process(p)

def uninstall():
  cloudlog.warning("uninstalling")
  with open('/cache/recovery/command', 'w') as f:
    f.write('--wipe_data\n')
  # IPowerManager.reboot(confirm=false, reason="recovery", wait=true)
  os.system("service call power 16 i32 0 s16 recovery i32 1")

def main():
  # the flippening!
  os.system('LD_LIBRARY_PATH="" content insert --uri content://settings/system --bind name:s:user_rotation --bind value:i:1')

  # disable bluetooth
  os.system('service call bluetooth_manager 8')

  if os.getenv("NOLOG") is not None:
    del managed_processes['loggerd']
    del managed_processes['tombstoned']
  if os.getenv("NOUPLOAD") is not None:
    del managed_processes['uploader']
  if os.getenv("NOVISION") is not None:
    del managed_processes['visiond']
  if os.getenv("LEAN") is not None:
    del managed_processes['uploader']
    del managed_processes['loggerd']
    del managed_processes['logmessaged']
    del managed_processes['logcatd']
    del managed_processes['tombstoned']
    del managed_processes['proclogd']
  if os.getenv("NOCONTROL") is not None:
    del managed_processes['controlsd']
    del managed_processes['plannerd']
    del managed_processes['radard']

  # support additional internal only extensions
  try:
    import selfdrive.manager_extensions
    selfdrive.manager_extensions.register(register_managed_process) # pylint: disable=no-member
  except ImportError:
    pass

  params = Params()
  params.manager_start()

  # set unset params
  if params.get("CompletedTrainingVersion") is None:
    params.put("CompletedTrainingVersion", "0")
  if params.get("IsMetric") is None:
    params.put("IsMetric", "0")
  if params.get("RecordFront") is None:
    params.put("RecordFront", "0")
  if params.get("HasAcceptedTerms") is None:
    params.put("HasAcceptedTerms", "0")
  if params.get("HasCompletedSetup") is None:
    params.put("HasCompletedSetup", "0")
  if params.get("IsUploadRawEnabled") is None:
    params.put("IsUploadRawEnabled", "1")
  if params.get("IsUploadVideoOverCellularEnabled") is None:
    params.put("IsUploadVideoOverCellularEnabled", "1")
  if params.get("IsGeofenceEnabled") is None:
    params.put("IsGeofenceEnabled", "-1")
  if params.get("SpeedLimitOffset") is None:
    params.put("SpeedLimitOffset", "0")
  if params.get("LongitudinalControl") is None:
    params.put("LongitudinalControl", "0")
  if params.get("LimitSetSpeed") is None:
    params.put("LimitSetSpeed", "0")
  if params.get("LimitSetSpeedNeural") is None:
    params.put("LimitSetSpeedNeural", "0")
  if params.get("LastUpdateTime") is None:
    t = datetime.datetime.now().isoformat()
    params.put("LastUpdateTime", t.encode('utf8'))
  if params.get("OpenpilotEnabledToggle") is None:
    params.put("OpenpilotEnabledToggle", "1")

  # is this chffrplus?
  if os.getenv("PASSIVE") is not None:
    params.put("Passive", str(int(os.getenv("PASSIVE"))))

  if params.get("Passive") is None:
    raise Exception("Passive must be set to continue")

  with Spinner() as spinner:
      spinner.update("0") # Show progress bar
      manager_update()
      manager_init()
      manager_prepare(spinner)

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
    cleanup_all_processes(None, None)

  if params.get("DoUninstall", encoding='utf8') == "1":
    uninstall()

if __name__ == "__main__":
  main()
  # manual exit because we are forked
  sys.exit(0)
