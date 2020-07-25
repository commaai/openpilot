#!/usr/bin/env python3
import time
import threading
import _thread
import signal
import sys

import cereal.messaging as messaging
from common.params import Params
import selfdrive.manager as manager
from selfdrive.test.helpers import set_params_enabled

def cputime_total(ct):
  return ct.cpuUser + ct.cpuSystem + ct.cpuChildrenUser + ct.cpuChildrenSystem


def print_cpu_usage(first_proc, last_proc):
  procs = [
    ("selfdrive.controls.controlsd", 59.46),
    ("selfdrive.locationd.locationd", 34.38),
    ("./loggerd", 33.90),
    ("selfdrive.controls.plannerd", 19.77),
    ("./_modeld", 12.74),
    ("selfdrive.locationd.paramsd", 11.53),
    ("selfdrive.controls.radard", 9.54),
    ("./_ui", 9.54),
    ("./camerad", 7.07),
    ("selfdrive.locationd.calibrationd", 6.81),
    ("./_sensord", 6.17),
    ("selfdrive.monitoring.dmonitoringd", 5.48),
    ("./boardd", 3.63),
    ("./_dmonitoringmodeld", 2.67),
    ("selfdrive.logmessaged", 2.71),
    ("selfdrive.thermald.thermald", 2.41),
    ("./proclogd", 1.54),
    ("./_gpsd", 0.09),
    ("./clocksd", 0.02),
    ("./ubloxd", 0.02),
    ("selfdrive.tombstoned", 0),
    ("./logcatd", 0),
  ]

  r = 0
  dt = (last_proc.logMonoTime - first_proc.logMonoTime) / 1e9
  result = "------------------------------------------------\n"
  for proc_name, normal_cpu_usage in procs:
    try:
      first = [p for p in first_proc.procLog.procs if proc_name in p.cmdline][0]
      last = [p for p in last_proc.procLog.procs if proc_name in p.cmdline][0]
      cpu_time = cputime_total(last) - cputime_total(first)
      cpu_usage = cpu_time / dt * 100.
      if cpu_usage > max(normal_cpu_usage * 1.1, normal_cpu_usage + 5.0):
        result += f"Warning {proc_name} using more CPU than normal\n"
        r = 1
      elif cpu_usage < min(normal_cpu_usage * 0.3, max(normal_cpu_usage - 1.0, 0.0)):
        result += f"Warning {proc_name} using less CPU than normal\n"
        r = 1
      result += f"{proc_name.ljust(35)}  {cpu_usage:.2f}%\n"
    except IndexError:
      result += f"{proc_name.ljust(35)}  NO METRICS FOUND\n"
      r = 1
  result += "------------------------------------------------\n"
  print(result)
  return r

def all_running():
  running = manager.get_running()
  return all(p in running and running[p].is_alive() for p in manager.car_started_processes)

return_code = 1
def test_thread():
  try:
    global return_code
    proc_sock = messaging.sub_sock('procLog', conflate=True, timeout=2000)

    # wait until everything's started and get first sample
    start_time = time.monotonic()
    while time.monotonic() - start_time < 120:
      if Params().get("CarParams") is not None:
        break
      time.sleep(2)
    first_proc = messaging.recv_sock(proc_sock, wait=True)
    if first_proc is None or not all_running():
      err_msg = "procLog recv timed out" if first_proc is None else "all car started process not running"
      print(f"\n\nTEST FAILED: {err_msg}\n\n")
      raise Exception

    # run for a minute and get last sample
    time.sleep(60)
    last_proc = messaging.recv_sock(proc_sock, wait=True)
    return_code = print_cpu_usage(first_proc, last_proc)
    if not all_running():
      return_code = 1
  finally:
    _thread.interrupt_main()

if __name__ == "__main__":


  # setup signal handler to exit with test status
  def handle_exit(sig, frame):
    sys.exit(return_code)
  signal.signal(signal.SIGINT, handle_exit)

  # start manager and test thread
  set_params_enabled()
  Params().delete("CarParams")

  t = threading.Thread(target=test_thread)
  t.daemon = True
  t.start()
  manager.main()
