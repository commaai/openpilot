import os
import sys

import zmq
from lru import LRU

from cereal import log
from common.realtime import Ratekeeper
import cereal.messaging as messaging
from cereal.services import service_list

def cputime_total(ct):
  return ct.user+ct.nice+ct.system+ct.idle+ct.iowait+ct.irq+ct.softirq

def cputime_busy(ct):
  return ct.user+ct.nice+ct.system+ct.irq+ct.softirq

def cpu_dtotal(l1, l2):
  t1_total = sum(cputime_total(ct) for ct in l1.cpuTimes)
  t2_total = sum(cputime_total(ct) for ct in l2.cpuTimes)
  return t2_total - t1_total

def cpu_percent(l1, l2):
  dtotal = cpu_dtotal(l1, l2)
  t1_busy = sum(cputime_busy(ct) for ct in l1.cpuTimes)
  t2_busy = sum(cputime_busy(ct) for ct in l2.cpuTimes)

  dbusy = t2_busy - t1_busy

  if dbusy < 0 or dtotal <= 0:
    return 0.0
  return dbusy / dtotal

def proc_cpu_percent(proc1, proc2, l1, l2):
  dtotal = cpu_dtotal(l1, l2)

  dproc = (proc2.cpuUser+proc2.cpuSystem) - (proc1.cpuUser+proc1.cpuSystem)
  if dproc < 0:
    return 0.0

  return dproc / dtotal

def display_cpu(pl1, pl2):
  l1, l2 = pl1.procLog, pl2.procLog

  print(cpu_percent(l1, l2))

  procs1 = dict((proc.pid, proc) for proc in l1.procs)
  procs2 = dict((proc.pid, proc) for proc in l2.procs)

  procs_print = 4

  procs_with_percent = sorted((proc_cpu_percent(procs1[proc.pid], proc, l1, l2), proc) for proc in l2.procs
                               if proc.pid in procs1)
  for percent, proc in procs_with_percent[-1:-procs_print-1:-1]:
    print(percent, proc.name)

  print()


def main():
  frame_cache = LRU(16)
  md_cache = LRU(16)
  plan_cache = LRU(16)

  frame_sock = messaging.sub_sock('frame')
  md_sock = messaging.sub_sock('model')
  plan_sock = messaging.sub_sock('plan')
  controls_state_sock = messaging.sub_sock('controlsState')

  proc = messaging.sub_sock('procLog')
  pls = [None, None]

  rk = Ratekeeper(10)
  while True:

    for msg in messaging.drain_sock(frame_sock):
      frame_cache[msg.frame.frameId] = msg

    for msg in messaging.drain_sock(md_sock):
      md_cache[msg.logMonoTime] = msg

    for msg in messaging.drain_sock(plan_sock):
      plan_cache[msg.logMonoTime] = msg

    controls_state = messaging.recv_sock(controls_state_sock)
    if controls_state is not None:
      plan_time = controls_state.controlsState.planMonoTime
      if plan_time != 0 and plan_time in plan_cache:
        plan = plan_cache[plan_time]
        md_time = plan.plan.mdMonoTime
        if md_time != 0 and md_time in md_cache:
          md = md_cache[md_time]
          frame_id = md.model.frameId
          if frame_id != 0 and frame_id in frame_cache:
            frame = frame_cache[frame_id]
            print("controls lag: %.2fms" % ((controls_state.logMonoTime - frame.frame.timestampEof) / 1e6))


    pls = (pls+messaging.drain_sock(proc))[-2:]
    if None not in pls:
      display_cpu(*pls)

    rk.keep_time()

if __name__ == "__main__":
  main()
