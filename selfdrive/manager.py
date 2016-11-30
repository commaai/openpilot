#!/usr/bin/env python
import os
import sys
import time
import importlib
import subprocess
import signal
import traceback
import usb1
from multiprocessing import Process
from common.services import service_list

import zmq

from setproctitle import setproctitle

from selfdrive.swaglog import cloudlog
import selfdrive.messaging as messaging
from selfdrive.thermal import read_thermal
from selfdrive.registration import register

import common.crash

# comment out anything you don't want to run
managed_processes = {
  "uploader": "selfdrive.loggerd.uploader",
  "controlsd": "selfdrive.controls.controlsd",
  "radard": "selfdrive.controls.radard",
  "calibrationd": "selfdrive.calibrationd.calibrationd",
  "loggerd": "selfdrive.loggerd.loggerd",
  "logmessaged": "selfdrive.logmessaged",
  #"boardd": "selfdrive.boardd.boardd",
  "logcatd": ("logcatd", ["./logcatd"]),
  "boardd": ("boardd", ["./boardd"]),   # switch to c++ boardd
  "ui": ("ui", ["./ui"]),
  "visiond": ("visiond", ["./visiond"]),
  "sensord": ("sensord", ["./sensord"]), }

running = {}

car_started_processes = ['controlsd', 'loggerd', 'visiond', 'sensord', 'radard', 'calibrationd']


# ****************** process management functions ******************
def launcher(proc, gctx):
  try:
    # unset the signals
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)

    # import the process
    mod = importlib.import_module(proc)

    # rename the process
    setproctitle(proc)

    # exec the process
    mod.main(gctx)
  except Exception:
    # can't install the crash handler becuase sys.excepthook doesn't play nice
    # with threads, so catch it here.
    common.crash.capture_exception()
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
    cwd = os.path.dirname(os.path.realpath(__file__))
    if pdir is not None:
      cwd = os.path.join(cwd, pdir)
    cloudlog.info("starting process %s" % name)
    running[name] = Process(name=name, target=nativelauncher, args=(pargs, cwd))
  running[name].start()

def kill_managed_process(name):
  if name not in running or name not in managed_processes:
    return
  cloudlog.info("killing %s" % name)
  running[name].terminate()
  running[name].join(5.0)
  if running[name].exitcode is None:
    cloudlog.info("killing %s with SIGKILL" % name)
    os.kill(running[name].pid, signal.SIGKILL)
    running[name].join()
    cloudlog.info("%s is finally dead" % name)
  else:
    cloudlog.info("%s is dead with %d" % (name, running[name].exitcode))
  del running[name]

def cleanup_all_processes(signal, frame):
  cloudlog.info("caught ctrl-c %s %s" % (signal, frame))
  for name in running.keys():
    kill_managed_process(name)
  sys.exit(0)

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
  os.environ['DONGLE_SECRET'] = dongle_secret

  cloudlog.bind_global(dongle_id=dongle_id)

  # set gctx
  gctx = {
    "calibration": {
      "initial_homography": [1.15728010e+00, -4.69379619e-02, 7.46450623e+01,
                             7.99253014e-02, 1.06372458e+00, 5.77762553e+01,
                             9.35543519e-05, -1.65429898e-04, 9.98062699e-01]
    }
  }

  # hook to kill all processes
  signal.signal(signal.SIGINT, cleanup_all_processes)
  signal.signal(signal.SIGTERM, cleanup_all_processes)

def manager_thread():
  # now loop
  context = zmq.Context()
  thermal_sock = messaging.pub_sock(context, service_list['thermal'].port)
  health_sock = messaging.sub_sock(context, service_list['health'].port)

  cloudlog.info("manager start")

  start_managed_process("logmessaged")
  start_managed_process("logcatd")
  start_managed_process("uploader")
  start_managed_process("ui")

  # *** wait for the board ***
  wait_for_device()

  # flash the device
  if os.getenv("NOPROG") is None:
    boarddir = os.path.dirname(os.path.abspath(__file__))+"/../board/"
    os.system("cd %s && make" % boarddir)

  start_managed_process("boardd")

  if os.getenv("STARTALL") is not None:
    for p in car_started_processes:
      start_managed_process(p)

  while 1:
    # get health of board, log this in "thermal"
    td = messaging.recv_sock(health_sock, wait=True)
    print td

    # replace thermald
    msg = read_thermal()
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

    # start constellation of processes when the car starts
    if td.health.started:
      for p in car_started_processes:
        start_managed_process(p)
    else:
      for p in car_started_processes:
        kill_managed_process(p)

    # check the status of all processes, did any of them die?
    for p in running:
      cloudlog.info("   running %s %s" % (p, running[p]))


# optional, build the c++ binaries and preimport the python for speed
def manager_prepare():
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

def manager_test():
  global managed_processes
  managed_processes = {}
  managed_processes["test1"] = ("test", ["./test.py"])
  managed_processes["test2"] = ("test", ["./test.py"])
  managed_processes["test3"] = "selfdrive.test.test"
  manager_prepare()
  start_managed_process("test1")
  start_managed_process("test2")
  start_managed_process("test3")
  print running
  time.sleep(3)
  kill_managed_process("test1")
  kill_managed_process("test2")
  kill_managed_process("test3")
  print running
  time.sleep(10)

def wait_for_device():
  while 1:
    try:
      context = usb1.USBContext()
      for device in context.getDeviceList(skip_on_error=True):
        if (device.getVendorID() == 0xbbaa and device.getProductID() == 0xddcc) or \
           (device.getVendorID() == 0x0483 and device.getProductID() == 0xdf11):
          handle = device.open()
          handle.claimInterface(0)
          cloudlog.info("found board")
          handle.close()
          return
    except Exception as e:
      print "exception", e,
    print "waiting..."
    time.sleep(1)

def main():
  if os.getenv("NOLOG") is not None:
    del managed_processes['loggerd']
  if os.getenv("NOBOARD") is not None:
    del managed_processes['boardd']

  manager_init()

  if len(sys.argv) > 1 and sys.argv[1] == "test":
    manager_test()
  else:
    manager_prepare()
    try:
      manager_thread()
    except Exception:
      traceback.print_exc()
      common.crash.capture_exception()
    finally:
      cleanup_all_processes(None, None)

if __name__ == "__main__":
  main()
