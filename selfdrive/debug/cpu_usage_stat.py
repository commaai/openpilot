#!/usr/bin/env python3
# type: ignore
'''
System tools like top/htop can only show current cpu usage values, so I write this script to do statistics jobs.
  Features:
    Use psutil library to sample cpu usage(avergage for all cores) of openpilot processes, at a rate of 5 samples/sec.
    Do cpu usage statistics periodically, 5 seconds as a cycle.
    Caculate the average cpu usage within this cycle.
    Caculate minumium/maximium/accumulated_average cpu usage as long term inspections.
    Monitor multiple processes simuteneously.
  Sample usage:
    root@localhost:/data/openpilot$ python selfdrive/debug/cpu_usage_stat.py boardd,ubloxd
    ('Add monitored proc:', './boardd')
    ('Add monitored proc:', 'python locationd/ubloxd.py')
    boardd: 1.96%, min: 1.96%, max: 1.96%, acc: 1.96%
    ubloxd.py: 0.39%, min: 0.39%, max: 0.39%, acc: 0.39%
'''
import psutil
import time
import os
import sys
import numpy as np
import argparse
import re
from collections import defaultdict


# Do statistics every 5 seconds
PRINT_INTERVAL = 5
SLEEP_INTERVAL = 0.2

monitored_proc_names = [
  'ubloxd', 'thermald', 'uploader', 'deleter', 'controlsd', 'plannerd', 'radard', 'mapd', 'loggerd', 'logmessaged', 'tombstoned',
  'logcatd', 'proclogd', 'boardd', 'pandad', './ui', 'ui', 'calibrationd', 'params_learner', 'modeld', 'dmonitoringmodeld', 'camerad', 'sensord', 'updated', 'gpsd', 'athena']
cpu_time_names = ['user', 'system', 'children_user', 'children_system']

timer = getattr(time, 'monotonic', time.time)


def get_arg_parser():
  parser = argparse.ArgumentParser(
    description="Unlogger and UI",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("proc_names", nargs="?", default='',
                      help="Process names to be monitored, comma seperated")
  parser.add_argument("--list_all", action='store_true',
                      help="Show all running processes' cmdline")
  parser.add_argument("--detailed_times", action='store_true',
                      help="show cpu time details (split by user, system, child user, child system)")
  return parser


if __name__ == "__main__":
  args = get_arg_parser().parse_args(sys.argv[1:])
  if args.list_all:
    for p in psutil.process_iter():
      print('cmdline', p.cmdline(), 'name', p.name())
    sys.exit(0)

  if len(args.proc_names) > 0:
    monitored_proc_names = args.proc_names.split(',')
  monitored_procs = []
  stats = {}
  for p in psutil.process_iter():
    if p == psutil.Process():
      continue
    matched = any([l for l in p.cmdline() if any([pn for pn in monitored_proc_names if re.match(r'.*{}.*'.format(pn), l, re.M | re.I)])])
    if matched:
      k = ' '.join(p.cmdline())
      print('Add monitored proc:', k)
      stats[k] = {'cpu_samples': defaultdict(list), 'min': defaultdict(lambda: None), 'max': defaultdict(lambda: None),
                  'avg': defaultdict(lambda: 0.0), 'last_cpu_times': None, 'last_sys_time': None}
      stats[k]['last_sys_time'] = timer()
      stats[k]['last_cpu_times'] = p.cpu_times()
      monitored_procs.append(p)
  i = 0
  interval_int = int(PRINT_INTERVAL / SLEEP_INTERVAL)
  while True:
    for p in monitored_procs:
      k = ' '.join(p.cmdline())
      cur_sys_time = timer()
      cur_cpu_times = p.cpu_times()
      cpu_times = np.subtract(cur_cpu_times, stats[k]['last_cpu_times']) / (cur_sys_time - stats[k]['last_sys_time'])
      stats[k]['last_sys_time'] = cur_sys_time
      stats[k]['last_cpu_times'] = cur_cpu_times
      cpu_percent = 0
      for num, name in enumerate(cpu_time_names):
        stats[k]['cpu_samples'][name].append(cpu_times[num])
        cpu_percent += cpu_times[num]
      stats[k]['cpu_samples']['total'].append(cpu_percent)
    time.sleep(SLEEP_INTERVAL)
    i += 1
    if i % interval_int == 0:
      l = []
      for k, stat in stats.items():
        if len(stat['cpu_samples']) <= 0:
          continue
        for name, samples in stat['cpu_samples'].items():
          samples = np.array(samples)
          avg = samples.mean()
          c = samples.size
          min_cpu = np.amin(samples)
          max_cpu = np.amax(samples)
          if stat['min'][name] is None or min_cpu < stat['min'][name]:
            stat['min'][name] = min_cpu
          if stat['max'][name] is None or max_cpu > stat['max'][name]:
            stat['max'][name] = max_cpu
          stat['avg'][name] = (stat['avg'][name] * (i - c) + avg * c) / (i)
          stat['cpu_samples'][name] = []

        msg = 'avg: {1:.2%}, min: {2:.2%}, max: {3:.2%} {0}'.format(os.path.basename(k), stat['avg']['total'], stat['min']['total'], stat['max']['total'])
        if args.detailed_times:
          for stat_type in ['avg', 'min', 'max']:
            msg += '\n {}: {}'.format(stat_type, [name + ':' + str(round(stat[stat_type][name]*100, 2)) for name in cpu_time_names])
        l.append((os.path.basename(k), stat['avg']['total'], msg))
      l.sort(key=lambda x: -x[1])
      for x in l:
        print(x[2])
      print('avg sum: {0:.2%} over {1} samples {2} seconds\n'.format(
        sum([stat['avg']['total'] for k, stat in stats.items()]), i, i * SLEEP_INTERVAL
      ))
