#!/usr/bin/env python3
import argparse
import capnp

from cereal.messaging import SubMaster
from common.numpy_fast import mean
from typing import Optional

def cputime_total(ct):
  return ct.user + ct.nice + ct.system + ct.idle + ct.iowait + ct.irq + ct.softirq


def cputime_busy(ct):
  return ct.user + ct.nice + ct.system + ct.irq + ct.softirq


def proc_cputime_total(ct):
  return ct.cpuUser + ct.cpuSystem + ct.cpuChildrenUser + ct.cpuChildrenSystem


def proc_name(proc):
  name = proc.name
  if len(proc.cmdline):
    name = proc.cmdline[0]
  if len(proc.exe):
    name = proc.exe + " - " + name

  return name


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--mem', action='store_true')
  parser.add_argument('--cpu', action='store_true')
  args = parser.parse_args()

  sm = SubMaster(['deviceState', 'procLog'])

  last_temp = 0.0
  last_mem = 0.0
  total_times = [0.]*8
  busy_times = [0.]*8

  prev_proclog: Optional[capnp._DynamicStructReader] = None
  prev_proclog_t: Optional[int] = None

  while True:
    sm.update()

    if sm.updated['deviceState']:
      t = sm['deviceState']
      last_temp = mean(t.cpuTempC)
      last_mem = t.memoryUsagePercent

    if sm.updated['procLog']:
      m = sm['procLog']

      cores = [0.]*8
      total_times_new = [0.]*8
      busy_times_new = [0.]*8

      for c in m.cpuTimes:
        n = c.cpuNum
        total_times_new[n] = cputime_total(c)
        busy_times_new[n] = cputime_busy(c)

      for n in range(8):
        t_busy = busy_times_new[n] - busy_times[n]
        t_total = total_times_new[n] - total_times[n]
        cores[n] = t_busy / t_total

      total_times = total_times_new[:]
      busy_times = busy_times_new[:]

      print(f"CPU {100.0 * mean(cores):.2f}% - RAM: {last_mem:.2f}% - Temp {last_temp:.2f}C")

      if args.cpu and prev_proclog is not None and prev_proclog_t is not None:
        procs = {}
        dt = (sm.logMonoTime['procLog'] - prev_proclog_t) / 1e9
        for proc in m.procs:
          try:
            name = proc_name(proc)
            prev_proc = [p for p in prev_proclog.procs if proc.pid == p.pid][0]
            cpu_time = proc_cputime_total(proc) - proc_cputime_total(prev_proc)
            cpu_usage = cpu_time / dt * 100.
            procs[name] = cpu_usage
          except IndexError:
            pass

        print("Top CPU usage:")
        for k, v in sorted(procs.items(), key=lambda item: item[1], reverse=True)[:10]:
          print(f"{k.rjust(70)}   {v:.2f} %")
        print()

      if args.mem:
        mems = {}
        for proc in m.procs:
          name = proc_name(proc)
          mems[name] = float(proc.memRss) / 1e6
        print("Top memory usage:")
        for k, v in sorted(mems.items(), key=lambda item: item[1], reverse=True)[:10]:
          print(f"{k.rjust(70)}   {v:.2f} MB")
        print()

      prev_proclog = m
      prev_proclog_t = sm.logMonoTime['procLog']
