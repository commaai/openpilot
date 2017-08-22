#!/usr/bin/env python
import os

# check if NEOS update is required
while 1:
  if ((not os.path.isfile("/VERSION")
      or int(open("/VERSION").read()) < 3)
      and not os.path.isfile("/data/media/0/noupdate")):
    os.system("curl -o /tmp/updater https://openpilot.comma.ai/updater && chmod +x /tmp/updater && /tmp/updater")
  else:
    break

import sys
import time
import importlib
import subprocess
import signal
import traceback
import usb1
from multiprocessing import Process
from selfdrive.services import service_list

import hashlib
import zmq

from setproctitle import setproctitle

from selfdrive.swaglog import cloudlog
import selfdrive.messaging as messaging
from selfdrive.thermal import read_thermal
from selfdrive.registration import register
from selfdrive.version import version

import common.crash as crash
from common.params import Params

from selfdrive.loggerd.config import ROOT

BASEDIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../")

# comment out anything you don't want to run
managed_processes = {
  "uploader": "selfdrive.loggerd.uploader",
  "controlsd": "selfdrive.controls.controlsd",
  "radard": "selfdrive.controls.radard",
  "loggerd": ("loggerd", ["./loggerd"]),
  "logmessaged": "selfdrive.logmessaged",
  "tombstoned": "selfdrive.tombstoned",
  "logcatd": ("logcatd", ["./logcatd"]),
  "proclogd": ("proclogd", ["./proclogd"]),
  "boardd": ("boardd", ["./boardd"]),   # switch to c++ boardd
  "ui": ("ui", ["./ui"]),
  "visiond": ("visiond", ["./visiond"]),
  "sensord": ("sensord", ["./sensord"]), }

running = {}
def get_running():
  return running

# due to qualcomm kernel bugs SIGKILLing visiond sometimes causes page table corruption
unkillable_processes = ['visiond']

# processes to end with SIGINT instead of SIGTERM
interrupt_processes = []

car_started_processes = [
  'controlsd',
  'loggerd',
  'sensord',
  'radard',
  'visiond',
  'proclogd',
]

def register_managed_process(name, desc, car_started=False):
  global managed_processes, car_started_processes
  print "registering", name
  managed_processes[name] = desc
  if car_started:
    car_started_processes.append(name)

# ****************** process management functions ******************
def launcher(proc, gctx):
  try:
    # import the process
    mod = importlib.import_module(proc)

    # rename the process
    setproctitle(proc)

    # exec the process
    mod.main(gctx)
  except KeyboardInterrupt:
    cloudlog.info("child %s got ctrl-c" % proc)
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
  if isinstance(proc, basestring):
    cloudlog.info("starting python %s" % proc)
    running[name] = Process(name=name, target=launcher, args=(proc, gctx))
  else:
    pdir, pargs = proc
    cwd = os.path.join(BASEDIR, "selfdrive")
    if pdir is not None:
      cwd = os.path.join(cwd, pdir)
    cloudlog.info("starting process %s" % name)
    running[name] = Process(name=name, target=nativelauncher, args=(pargs, cwd))
  running[name].start()

def kill_managed_process(name):
  if name not in running or name not in managed_processes:
    return
  cloudlog.info("killing %s" % name)

  if running[name].exitcode is None:
    if name in interrupt_processes:
      os.kill(running[name].pid, signal.SIGINT)
    else:
      running[name].terminate()

    # give it 5 seconds to die
    running[name].join(5.0)
    if running[name].exitcode is None:
      if name in unkillable_processes:
        cloudlog.critical("unkillable process %s failed to exit! rebooting in 15 if it doesn't die" % name)
        running[name].join(15.0)
        if running[name].exitcode is None:
          cloudlog.critical("FORCE REBOOTING PHONE!")
          os.system("date > /sdcard/unkillable_reboot")
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
  manage_baseui(False)
  for name in running.keys():
    kill_managed_process(name)
  sys.exit(0)

baseui_running = False
def manage_baseui(start):
  global baseui_running
  if start and not baseui_running:
    cloudlog.info("starting baseui")
    os.system("am start -n com.baseui/.MainActivity")
    baseui_running = True
  elif not start and baseui_running:
    cloudlog.info("stopping baseui")
    os.system("am force-stop com.baseui")
    baseui_running = False

# ****************** run loop ******************

def manager_init():
  global gctx

  reg_res = register()
  if reg_res:
    dongle_id, dongle_secret = reg_res
  else:
    raise Exception("server registration failed")

  # set dongle id
  cloudlog.info("dongle id is " + dongle_id)
  os.environ['DONGLE_ID'] = dongle_id

  dirty = subprocess.call(["git", "diff-index", "--quiet", "origin/release", "--"]) != 0
  cloudlog.info("dirty is %d" % dirty)
  if not dirty:
    os.environ['CLEAN'] = '1'

  cloudlog.bind_global(dongle_id=dongle_id, version=version, dirty=dirty)
  crash.bind_user(id=dongle_id)
  crash.bind_extra(version=version, dirty=dirty)

  os.umask(0)
  try:
    os.mkdir(ROOT, 0777)
  except OSError:
    pass

  # set gctx
  gctx = {}

def system(cmd):
  try:
    cloudlog.info("running %s" % cmd)
    subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
  except subprocess.CalledProcessError, e:
    cloudlog.event("running failed",
      cmd=e.cmd,
      output=e.output,
      returncode=e.returncode)

def manager_thread():
  global baseui_running

  # now loop
  context = zmq.Context()
  thermal_sock = messaging.pub_sock(context, service_list['thermal'].port)
  health_sock = messaging.sub_sock(context, service_list['health'].port)

  cloudlog.info("manager start")
  cloudlog.info(dict(os.environ))

  start_managed_process("logmessaged")
  start_managed_process("logcatd")
  start_managed_process("tombstoned")
  start_managed_process("uploader")
  start_managed_process("ui")
  manage_baseui(True)

  panda = False
  if os.getenv("NOBOARD") is None:
    # *** wait for the board ***
    panda = wait_for_device() == 0x2300

  # flash the device
  if os.getenv("NOPROG") is None:
    # flash the board
    boarddir = os.path.join(BASEDIR, "panda/board/")
    mkfile = "Makefile" if panda else "Makefile.legacy"
    print "using", mkfile
    system("cd %s && make -f %s" % (boarddir, mkfile))

  start_managed_process("boardd")

  started = False
  logger_dead = False
  count = 0

  # set 5 second timeout on health socket
  # 5x slower than expected
  health_sock.RCVTIMEO = 5000

  while 1:
    # get health of board, log this in "thermal"
    td = messaging.recv_sock(health_sock, wait=True)
    print td

    # replace thermald
    msg = read_thermal()

    # loggerd is gated based on free space
    statvfs = os.statvfs(ROOT)
    avail = (statvfs.f_bavail * 1.0)/statvfs.f_blocks

    # thermal message now also includes free space
    msg.thermal.freeSpace = avail
    with open("/sys/class/power_supply/battery/capacity") as f:
      msg.thermal.batteryPercent = int(f.read())
    with open("/sys/class/power_supply/battery/status") as f:
      msg.thermal.batteryStatus = f.read().strip()
    thermal_sock.send(msg.to_bytes())
    print msg

    # TODO: add car battery voltage check
    max_temp = max(msg.thermal.cpu0, msg.thermal.cpu1,
                   msg.thermal.cpu2, msg.thermal.cpu3) / 10.0

    # uploader is gated based on the phone temperature
    if max_temp > 85.0:
      cloudlog.info("over temp: %r", max_temp)
      kill_managed_process("uploader")
    elif max_temp < 70.0:
      start_managed_process("uploader")

    if avail < 0.05:
      logger_dead = True

    # start constellation of processes when the car starts
    # with 2% left, we killall, otherwise the phone is bricked
    if td is not None and td.health.started and avail > 0.02:
      if not started:
        Params().car_start()
        started = True
      for p in car_started_processes:
        if p == "loggerd" and logger_dead:
          kill_managed_process(p)
        else:
          start_managed_process(p)
      manage_baseui(False)
    else:
      manage_baseui(True)
      started = False
      logger_dead = False
      for p in car_started_processes:
        kill_managed_process(p)

      # shutdown if the battery gets lower than 10%, we aren't running, and we are discharging
      if msg.thermal.batteryPercent < 5 and msg.thermal.batteryStatus == "Discharging":
        os.system('LD_LIBRARY_PATH="" svc power shutdown')

    # check the status of baseui
    baseui_running = 'com.baseui' in subprocess.check_output(["ps"])

    # check the status of all processes, did any of them die?
    for p in running:
      cloudlog.debug("   running %s %s" % (p, running[p]))

    # report to server once per minute
    if (count%60) == 0:
      cloudlog.event("STATUS_PACKET",
        running=running.keys(),
        count=count,
        health=(td.to_dict() if td else None),
        thermal=msg.to_dict())

    count += 1

def get_installed_apks():
  dat = subprocess.check_output(["pm", "list", "packages", "-3", "-f"]).strip().split("\n")
  ret = {}
  for x in dat:
    if x.startswith("package:"):
      v,k = x.split("package:")[1].split("=")
      ret[k] = v
  return ret

# optional, build the c++ binaries and preimport the python for speed
def manager_prepare():
  # update submodules
  system("cd %s && git submodule init panda opendbc pyextra" % BASEDIR)
  system("cd %s && git submodule update panda opendbc pyextra" % BASEDIR)

  # build cereal first
  subprocess.check_call(["make", "-j4"], cwd=os.path.join(BASEDIR, "cereal"))

  # build all processes
  os.chdir(os.path.dirname(os.path.abspath(__file__)))
  for p in managed_processes:
    proc = managed_processes[p]
    if isinstance(proc, basestring):
      # import this python
      cloudlog.info("preimporting %s" % proc)
      importlib.import_module(proc)
    else:
      # build this process
      cloudlog.info("building %s" % (proc,))
      try:
        subprocess.check_call(["make", "-j4"], cwd=proc[0])
      except subprocess.CalledProcessError:
        # make clean if the build failed
        cloudlog.info("building %s failed, make clean" % (proc, ))
        subprocess.check_call(["make", "clean"], cwd=proc[0])
        subprocess.check_call(["make", "-j4"], cwd=proc[0])

  # install apks
  installed = get_installed_apks()
  for app in os.listdir(os.path.join(BASEDIR, "apk/")):
    if ".apk" in app:
      app = app.split(".apk")[0]
      if app not in installed:
        installed[app] = None
  cloudlog.info("installed apks %s" % (str(installed), ))
  for app in installed:
    apk_path = os.path.join(BASEDIR, "apk/"+app+".apk")
    if os.path.isfile(apk_path):
      h1 = hashlib.sha1(open(apk_path).read()).hexdigest()
      h2 = None
      if installed[app] is not None:
        h2 = hashlib.sha1(open(installed[app]).read()).hexdigest()
        cloudlog.info("comparing version of %s  %s vs %s" % (app, h1, h2))
      if h2 is None or h1 != h2:
        cloudlog.info("installing %s" % app)
        for do_uninstall in [False, True]:
          if do_uninstall:
            cloudlog.info("needing to uninstall %s" % app)
            os.system("pm uninstall %s" % app)
          ret = os.system("cp %s /sdcard/%s.apk && pm install -r /sdcard/%s.apk && rm /sdcard/%s.apk" % (apk_path, app, app, app))
          if ret == 0:
            break
        assert ret == 0

def wait_for_device():
  while 1:
    try:
      context = usb1.USBContext()
      for device in context.getDeviceList(skip_on_error=True):
        if (device.getVendorID() == 0xbbaa and device.getProductID() == 0xddcc) or \
           (device.getVendorID() == 0x0483 and device.getProductID() == 0xdf11):
          bcd = device.getbcdDevice()
          handle = device.open()
          handle.claimInterface(0)
          cloudlog.info("found board")
          handle.close()
          return bcd
    except Exception as e:
      print "exception", e,
    print "waiting..."
    time.sleep(1)

def main():
  if os.getenv("NOLOG") is not None:
    del managed_processes['loggerd']
    del managed_processes['tombstoned']
  if os.getenv("NOUPLOAD") is not None:
    del managed_processes['uploader']
  if os.getenv("NOVISION") is not None:
    del managed_processes['visiond']
  if os.getenv("NOBOARD") is not None:
    del managed_processes['boardd']
  if os.getenv("LEAN") is not None:
    del managed_processes['uploader']
    del managed_processes['loggerd']
    del managed_processes['logmessaged']
    del managed_processes['logcatd']
    del managed_processes['tombstoned']
    del managed_processes['proclogd']
  if os.getenv("NOCONTROL") is not None:
    del managed_processes['controlsd']
    del managed_processes['radard']

  # support additional internal only extensions
  try:
    import selfdrive.manager_extensions
    selfdrive.manager_extensions.register(register_managed_process)
  except ImportError:
    pass

  params = Params()
  params.manager_start()

  # set unset params
  if params.get("IsMetric") is None:
    params.put("IsMetric", "0")
  if params.get("IsRearViewMirror") is None:
    params.put("IsRearViewMirror", "1")

  # put something on screen while we set things up
  if os.getenv("PREPAREONLY") is not None:
    spinner_proc = None
  else:
    spinner_proc = subprocess.Popen(["./spinner", "loading openpilot..."],
      cwd=os.path.join(BASEDIR, "selfdrive", "ui", "spinner"),
      close_fds=True)
  try:
    manager_init()
    manager_prepare()
  finally:
    if spinner_proc:
      spinner_proc.terminate()

  if os.getenv("PREPAREONLY") is not None:
    sys.exit(0)

  try:
    manager_thread()
  except Exception:
    traceback.print_exc()
    crash.capture_exception()
  finally:
    cleanup_all_processes(None, None)

if __name__ == "__main__":
  main()
