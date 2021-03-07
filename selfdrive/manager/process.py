import importlib
import os
import signal
import time
import subprocess
from abc import ABC, abstractmethod
from multiprocessing import Process

from setproctitle import setproctitle  # pylint: disable=no-name-in-module

import cereal.messaging as messaging
import selfdrive.crash as crash
from common.basedir import BASEDIR
from common.params import Params
from selfdrive.swaglog import cloudlog
from selfdrive.hardware import HARDWARE
from cereal import log


def launcher(proc):
  try:
    # import the process
    mod = importlib.import_module(proc)

    # rename the process
    setproctitle(proc)

    # create new context since we forked
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
  os.execvp(pargs[0], pargs)


def join_process(process, timeout):
  # Process().join(timeout) will hang due to a python 3 bug: https://bugs.python.org/issue28382
  # We have to poll the exitcode instead
  t = time.monotonic()
  while time.monotonic() - t < timeout and process.exitcode is None:
    time.sleep(0.001)


class ManagerProcess(ABC):
  unkillable = False
  daemon = False
  sigkill = False
  proc = None
  name = ""

  @abstractmethod
  def prepare(self):
    pass

  @abstractmethod
  def start(self):
    pass

  def stop(self, retry=True):
    if self.proc is None:
      return

    cloudlog.info(f"killing {self.name}")

    if self.proc.exitcode is None:
      sig = signal.SIGKILL if self.sigkill else signal.SIGINT
      self.signal(sig)

      join_process(self.proc, 5)

      # If process failed to die send SIGKILL or reboot
      if self.proc.exitcode is None and retry:
        if self.unkillable:
          cloudlog.critical(f"unkillable process {self.name} failed to exit! rebooting in 15 if it doesn't die")
          join_process(self.proc, 15)

          if self.proc.exitcode is None:
            cloudlog.critical(f"unkillable process {self.name} failed to die!")
            os.system("date >> /data/unkillable_reboot")
            os.sync()
            HARDWARE.reboot()
            raise RuntimeError
        else:
          cloudlog.info(f"killing {self.name} with SIGKILL")
          self.signal(signal.SIGKILL)
          self.proc.join()

    ret = self.proc.exitcode
    cloudlog.info(f"{self.name} is dead with {ret}")

    if self.proc.exitcode is not None:
      self.proc = None

    return ret

  def signal(self, sig):
    if self.proc.exitcode is not None and self.proc.pid is not None:
      return

    cloudlog.info(f"sending signal {sig} to {self.name}")
    os.kill(self.proc.pid, sig)

  def get_process_state_msg(self):
    state = log.ManagerState.ProcessState.new_message()
    state.name = self.name
    if self.proc:
      state.running = self.proc.is_alive()
      state.pid = self.proc.pid or 0
      state.exitCode = self.proc.exitcode or 0
    return state


class NativeProcess(ManagerProcess):
  def __init__(self, name, cwd, cmdline, persistent=False, driverview=False, unkillable=False, sigkill=False):
    self.name = name
    self.cwd = cwd
    self.cmdline = cmdline
    self.persistent = persistent
    self.driverview = driverview
    self.unkillable = unkillable
    self.sigkill = sigkill

  def prepare(self):
    pass

  def start(self):
    if self.proc is not None:
      return

    cwd = os.path.join(BASEDIR, self.cwd)
    cloudlog.info("starting process %s" % self.name)
    self.proc = Process(name=self.name, target=nativelauncher, args=(self.cmdline, cwd))
    self.proc.start()


class PythonProcess(ManagerProcess):
  def __init__(self, name, module, persistent=False, driverview=False, unkillable=False, sigkill=False):
    self.name = name
    self.module = module
    self.persistent = persistent
    self.driverview = driverview
    self.unkillable = unkillable
    self.sigkill = sigkill

  def prepare(self):
    cloudlog.info("preimporting %s" % self.module)
    importlib.import_module(self.module)

  def start(self):
    if self.proc is not None:
      return

    cloudlog.info("starting python %s" % self.module)
    self.proc = Process(name=self.name, target=launcher, args=(self.module,))
    self.proc.start()


class DaemonProcess(ManagerProcess):
  """Python process that has to stay running accross manager restart.
  This is used for athena so you don't lose SSH access when restarting manager."""
  def __init__(self, name, module, param_name):
    self.name = name
    self.module = module
    self.param_name = param_name
    self.persistent = True

  def prepare(self):
    pass

  def start(self):
    params = Params()
    pid = params.get(self.param_name, encoding='utf-8')

    if pid is not None:
      try:
        os.kill(int(pid), 0)
        with open(f'/proc/{pid}/cmdline') as f:
          if self.module in f.read():
            # daemon is running
            return
      except (OSError, FileNotFoundError):
        # process is dead
        pass

    cloudlog.info("starting daemon %s" % self.name)
    proc = subprocess.Popen(['python', '-m', self.module],  # pylint: disable=subprocess-popen-preexec-fn
                               stdin=open('/dev/null', 'r'),
                               stdout=open('/dev/null', 'w'),
                               stderr=open('/dev/null', 'w'),
                               preexec_fn=os.setpgrp)

    params.put(self.param_name, str(proc.pid))

  def stop(self, retry=True):
    pass


def ensure_running(procs, started, driverview=False, not_run=None):
  if not_run is None:
    not_run = []

  # TODO: can we do this in parallel?
  for p in procs:
    if p.name in not_run:
      p.stop()
    elif p.persistent:
      p.start()
    elif p.driverview and driverview:
      p.start()
    elif started:
      p.start()
    else:
      p.stop()
