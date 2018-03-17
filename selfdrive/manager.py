#!/usr/bin/env python2.7
import os
import sys
import fcntl
import errno
import signal

if __name__ == "__main__":
  if os.path.isfile("/init.qcom.rc") \
      and (not os.path.isfile("/VERSION") or int(open("/VERSION").read()) < 4):
    raise Exception("NEOS outdated")

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
        sys.stdout.write(dat)
      except (OSError, IOError):
        pass

    os._exit(os.wait()[1])

import glob
import shutil
import hashlib
import importlib
import subprocess
import traceback
from multiprocessing import Process

from common.basedir import BASEDIR
sys.path.append(os.path.join(BASEDIR, "pyextra"))
os.environ['BASEDIR'] = BASEDIR

import zmq
from setproctitle import setproctitle

from common.params import Params
from common.realtime import sec_since_boot

from selfdrive.services import service_list
from selfdrive.swaglog import cloudlog
import selfdrive.messaging as messaging
from selfdrive.thermal import read_thermal
from selfdrive.registration import register
from selfdrive.version import version, dirty
import selfdrive.crash as crash

from selfdrive.loggerd.config import ROOT

EON = os.path.exists("/EON")

# comment out anything you don't want to run
managed_processes = {
  "uploader": "selfdrive.loggerd.uploader",
  "controlsd": "selfdrive.controls.controlsd",
  "radard": "selfdrive.controls.radard",
  "ubloxd": "selfdrive.locationd.ubloxd",
  "locationd_dummy": "selfdrive.locationd.locationd_dummy",
  "loggerd": ("selfdrive/loggerd", ["./loggerd"]),
  "logmessaged": "selfdrive.logmessaged",
  "tombstoned": "selfdrive.tombstoned",
  "logcatd": ("selfdrive/logcatd", ["./logcatd"]),
  "proclogd": ("selfdrive/proclogd", ["./proclogd"]),
  "boardd": ("selfdrive/boardd", ["./boardd"]),   # not used directly
  "pandad": "selfdrive.pandad",
  "ui": ("selfdrive/ui", ["./ui"]),
  "visiond": ("selfdrive/visiond", ["./visiond"]),
  "sensord": ("selfdrive/sensord", ["./sensord"]),
  "gpsd": ("selfdrive/sensord", ["./gpsd"]),
  "updated": "selfdrive.updated",
  #"gpsplanner": "selfdrive.controls.gps_plannerd",
}

running = {}
def get_running():
  return running

# due to qualcomm kernel bugs SIGKILLing visiond sometimes causes page table corruption
unkillable_processes = ['visiond']

# processes to end with SIGINT instead of SIGTERM
interrupt_processes = []

persistent_processes = [
  'logmessaged',
  'logcatd',
  'tombstoned',
  'uploader',
  'ui',
  'gpsd',
  'ubloxd',
  'locationd_dummy',
  'updated',
]

car_started_processes = [
  'controlsd',
  'loggerd',
  'sensord',
  'radard',
  'visiond',
  'proclogd',
  # 'gpsplanner,
]

def register_managed_process(name, desc, car_started=False):
  global managed_processes, car_started_processes, persistent_processes
  print "registering", name
  managed_processes[name] = desc
  if car_started:
    car_started_processes.append(name)
  else:
    persistent_processes.append(name)

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
  if isinstance(proc, basestring):
    cloudlog.info("starting python %s" % proc)
    running[name] = Process(name=name, target=launcher, args=(proc, gctx))
  else:
    pdir, pargs = proc
    cwd = os.path.join(BASEDIR, pdir)
    cloudlog.info("starting process %s" % name)
    running[name] = Process(name=name, target=nativelauncher, args=(pargs, cwd))
  running[name].start()

def prepare_managed_process(p):
  proc = managed_processes[p]
  if isinstance(proc, basestring):
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

  for p in ("com.waze", "com.spotify.music", "ai.comma.plus.offroad", "ai.comma.plus.frame"):
    system("am force-stop %s" % p)

  for name in running.keys():
    kill_managed_process(name)
  cloudlog.info("everything is dead")


# ****************** run loop ******************

def manager_init(should_register=True):
  global gctx

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

  cloudlog.bind_global(dongle_id=dongle_id, version=version, dirty=dirty, is_eon=EON)
  crash.bind_user(id=dongle_id)
  crash.bind_extra(version=version, dirty=dirty, is_eon=EON)

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
      output=e.output[-1024:],
      returncode=e.returncode)

def setup_eon_fan():
  if not EON:
    return

  os.system("echo 2 > /sys/module/dwc3_msm/parameters/otg_switch")

  from smbus2 import SMBus
  bus = SMBus(7, force=True)
  bus.write_byte_data(0x21, 0x10, 0xf)   # mask all interrupts
  bus.write_byte_data(0x21, 0x03, 0x1)   # set drive current and global interrupt disable
  bus.write_byte_data(0x21, 0x02, 0x2)   # needed?
  bus.write_byte_data(0x21, 0x04, 0x4)   # manual override source
  bus.close()

last_eon_fan_val = None
def set_eon_fan(val):
  global last_eon_fan_val

  if not EON:
    return

  from smbus2 import SMBus
  if last_eon_fan_val is None or last_eon_fan_val != val:
    bus = SMBus(7, force=True)
    bus.write_byte_data(0x21, 0x04, 0x2)
    bus.write_byte_data(0x21, 0x03, (val*2)+1)
    bus.write_byte_data(0x21, 0x04, 0x4)
    bus.close()
    last_eon_fan_val = val


# temp thresholds to control fan speed - high hysteresis
_TEMP_THRS_H = [50., 65., 80., 10000]
# temp thresholds to control fan speed - low hysteresis
_TEMP_THRS_L = [42.5, 57.5, 72.5, 10000]
# fan speed options
_FAN_SPEEDS = [0, 16384, 32768, 65535]
# max fan speed only allowed if battery if hot
_BAT_TEMP_THERSHOLD = 45.

def handle_fan(max_temp, bat_temp, fan_speed):
  new_speed_h = next(speed for speed, temp_h in zip(_FAN_SPEEDS, _TEMP_THRS_H) if temp_h > max_temp)
  new_speed_l = next(speed for speed, temp_l in zip(_FAN_SPEEDS, _TEMP_THRS_L) if temp_l > max_temp)

  if new_speed_h > fan_speed:
    # update speed if using the high thresholds results in fan speed increment
    fan_speed = new_speed_h
  elif new_speed_l < fan_speed:
    # update speed if using the low thresholds results in fan speed decrement
    fan_speed = new_speed_l

  if bat_temp < _BAT_TEMP_THERSHOLD:
    # no max fan speed unless battery is hot
    fan_speed = min(fan_speed, _FAN_SPEEDS[-2])

  set_eon_fan(fan_speed/16384)

  return fan_speed

class LocationStarter(object):
  def __init__(self):
    self.last_good_loc = 0
  def update(self, started_ts, location):
    rt = sec_since_boot()

    if location is None or location.accuracy > 50 or location.speed < 2:
      # bad location, stop if we havent gotten a location in a while
      # dont stop if we're been going for less than a minute
      if started_ts:
        if rt-self.last_good_loc > 60. and rt-started_ts > 60:
          cloudlog.event("location_stop",
            ts=rt,
            started_ts=started_ts,
            last_good_loc=self.last_good_loc,
            location=location.to_dict() if location else None)
          return False
        else:
          return True
      else:
        return False

    self.last_good_loc = rt

    if started_ts:
      return True
    else:
      cloudlog.event("location_start", location=location.to_dict() if location else None)
      return location.speed*3.6 > 10

def manager_thread():
  # now loop
  context = zmq.Context()
  thermal_sock = messaging.pub_sock(context, service_list['thermal'].port)
  health_sock = messaging.sub_sock(context, service_list['health'].port)
  location_sock = messaging.sub_sock(context, service_list['gpsLocation'].port)

  cloudlog.info("manager start")
  cloudlog.info({"environ": os.environ})

  for p in persistent_processes:
    start_managed_process(p)

  # start frame
  system("am start -n ai.comma.plus.frame/.MainActivity")

  # do this before panda flashing
  setup_eon_fan()

  if os.getenv("NOBOARD") is None:
    start_managed_process("pandad")

  params = Params()

  passive_starter = LocationStarter()

  started_ts = None
  logger_dead = False
  count = 0
  fan_speed = 0
  ignition_seen = False
  battery_was_high = False
  panda_seen = False

  health_sock.RCVTIMEO = 1500

  while 1:
    # get health of board, log this in "thermal"
    td = messaging.recv_sock(health_sock, wait=True)
    location = messaging.recv_sock(location_sock)

    location = location.gpsLocation if location else None

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
    with open("/sys/class/power_supply/usb/online") as f:
      msg.thermal.usbOnline = bool(int(f.read()))

    # TODO: add car battery voltage check
    max_temp = max(msg.thermal.cpu0, msg.thermal.cpu1,
                   msg.thermal.cpu2, msg.thermal.cpu3) / 10.0
    bat_temp = msg.thermal.bat/1000.
    fan_speed = handle_fan(max_temp, bat_temp, fan_speed)
    msg.thermal.fanSpeed = fan_speed

    msg.thermal.started = started_ts is not None
    msg.thermal.startedTs = int(1e9*(started_ts or 0))

    thermal_sock.send(msg.to_bytes())
    print msg

    # uploader is gated based on the phone temperature
    if max_temp > 85.0:
      cloudlog.warning("over temp: %r", max_temp)
      kill_managed_process("uploader")
    elif max_temp < 70.0:
      start_managed_process("uploader")

    if avail < 0.05:
      logger_dead = True

    # start constellation of processes when the car starts
    ignition = td is not None and td.health.started
    ignition_seen = ignition_seen or ignition

    # add voltage check for ignition
    if not ignition_seen and td is not None and td.health.voltage > 13500:
      ignition = True

    do_uninstall = params.get("DoUninstall") == "1"
    accepted_terms = params.get("HasAcceptedTerms") == "1"

    should_start = ignition

    # have we seen a panda?
    panda_seen = panda_seen or td is not None

    # start on gps if we have no connection to a panda
    if not panda_seen:
      should_start = should_start or passive_starter.update(started_ts, location)

    # with 2% left, we killall, otherwise the phone will take a long time to boot
    should_start = should_start and avail > 0.02

    # require usb power
    should_start = should_start and msg.thermal.usbOnline

    should_start = should_start and accepted_terms and (not do_uninstall)

    # if any CPU gets above 107 or the battery gets above 63, kill all processes
    # controls will warn with CPU above 95 or battery above 60
    if max_temp > 107.0 or msg.thermal.bat >= 63000:
      # TODO: Add a better warning when this is happening
      should_start = False

    if should_start:
      if not started_ts:
        params.car_start()
        started_ts = sec_since_boot()
      for p in car_started_processes:
        if p == "loggerd" and logger_dead:
          kill_managed_process(p)
        else:
          start_managed_process(p)
    else:
      started_ts = None
      logger_dead = False
      for p in car_started_processes:
        kill_managed_process(p)

      # shutdown if the battery gets lower than 5%, we aren't running, and we are discharging
      if msg.thermal.batteryPercent < 5 and msg.thermal.batteryStatus == "Discharging" and battery_was_high:
        os.system('LD_LIBRARY_PATH="" svc power shutdown')
      if msg.thermal.batteryPercent > 10:
        battery_was_high = True

    # check the status of all processes, did any of them die?
    for p in running:
      cloudlog.debug("   running %s %s" % (p, running[p]))

    # report to server once per minute
    if (count%60) == 0:
      cloudlog.event("STATUS_PACKET",
        running=running.keys(),
        count=count,
        health=(td.to_dict() if td else None),
        location=(location.to_dict() if location else None),
        thermal=msg.to_dict())

    if do_uninstall:
      break

    count += 1

def get_installed_apks():
  dat = subprocess.check_output(["pm", "list", "packages", "-f"]).strip().split("\n")
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
  # patch apks
  if os.getenv("PREPAREONLY"):
    # assume we have internet, download too
    patched = subprocess.call([os.path.join(BASEDIR, "apk/external/patcher.py")])
  else:
    patched = subprocess.call([os.path.join(BASEDIR, "apk/external/patcher.py"), "patch"])
  cloudlog.info("patcher: %r" % (patched,))

  # install apks
  installed = get_installed_apks()

  install_apks = (glob.glob(os.path.join(BASEDIR, "apk/*.apk"))
                  + glob.glob(os.path.join(BASEDIR, "apk/external/out/*.apk")))
  for apk in install_apks:
    app = os.path.basename(apk)[:-4]
    if app not in installed:
      installed[app] = None

  cloudlog.info("installed apks %s" % (str(installed), ))

  for app in installed.iterkeys():

    apk_path = os.path.join(BASEDIR, "apk/"+app+".apk")
    if not os.path.exists(apk_path):
      apk_path = os.path.join(BASEDIR, "apk/external/out/"+app+".apk")
    if not os.path.exists(apk_path):
      continue

    h1 = hashlib.sha1(open(apk_path).read()).hexdigest()
    h2 = None
    if installed[app] is not None:
      h2 = hashlib.sha1(open(installed[app]).read()).hexdigest()
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

def manager_prepare():

  # build cereal first
  subprocess.check_call(["make", "-j4"], cwd=os.path.join(BASEDIR, "cereal"))

  # build all processes
  os.chdir(os.path.dirname(os.path.abspath(__file__)))
  for p in managed_processes:
    prepare_managed_process(p)

def uninstall():
  cloudlog.warning("uninstalling")
  with open('/cache/recovery/command', 'w') as f:
    f.write('--wipe_data\n')
  # IPowerManager.reboot(confirm=false, reason="recovery", wait=true)
  os.system("service call power 16 i32 0 s16 recovery i32 1")

def main():
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
    params.put("IsRearViewMirror", "0")
  if params.get("IsFcwEnabled") is None:
    params.put("IsFcwEnabled", "1")
  if params.get("HasAcceptedTerms") is None:
    params.put("HasAcceptedTerms", "0")
  if params.get("IsUploadVideoOverCellularEnabled") is None:
    params.put("IsUploadVideoOverCellularEnabled", "1")

  # is this chffrplus?
  if os.getenv("PASSIVE") is not None:
    params.put("Passive", str(int(os.getenv("PASSIVE"))))

  if params.get("Passive") is None:
    raise Exception("Passive must be set to continue")

  # put something on screen while we set things up
  if os.getenv("PREPAREONLY") is not None:
    spinner_proc = None
  else:
    spinner_proc = subprocess.Popen(["./spinner", "loading..."],
      cwd=os.path.join(BASEDIR, "selfdrive", "ui", "spinner"),
      close_fds=True)
  try:
    manager_update()
    manager_init()
    manager_prepare()
  finally:
    if spinner_proc:
      spinner_proc.terminate()

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

  if params.get("DoUninstall") == "1":
    uninstall()

if __name__ == "__main__":
  main()
  # manual exit because we are forked
  sys.exit(0)
