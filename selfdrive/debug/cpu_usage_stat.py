import psutil
import time
import os
import sys
import numpy as np
import argparse
import re

'''
System tools like top/htop can only show current cpu usage values, so I write this script to do statistics jobs.
  Features:
    Use psutil library to sample cpu usage(avergage for all cores) of OpenPilot processes, at a rate of 5 samples/sec.
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

# Do statistics every 5 seconds
PRINT_INTERVAL = 5
SLEEP_INTERVAL = 0.2

monitored_proc_names = [
  'ubloxd', 'thermald', 'uploader', 'controlsd', 'plannerd', 'radard', 'mapd', 'loggerd' , 'logmessaged', 'tombstoned',
  'logcatd', 'proclogd', 'boardd', 'pandad', './ui', 'calibrationd', 'locationd', 'visiond', 'sensord', 'updated', 'gpsd', 'athena']


def get_arg_parser():
  parser = argparse.ArgumentParser(
    description="Unlogger and UI",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("proc_names", nargs="?", default='',
                      help="Process names to be monitored, comma seperated")
  parser.add_argument("--list_all", nargs="?", type=bool, default=False,
                      help="Show all running processes' cmdline")
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
      stats[k] = {'cpu_samples': [], 'avg_cpu': None, 'min': None, 'max': None}
      monitored_procs.append(p)
  i = 0
  interval_int = int(PRINT_INTERVAL / SLEEP_INTERVAL)
  while True:
    for p in monitored_procs:
      k = ' '.join(p.cmdline())
      stats[k]['cpu_samples'].append(p.cpu_percent())
    time.sleep(SLEEP_INTERVAL)
    i += 1
    if i % interval_int == 0:
      l = []
      avg_cpus = []
      for k, stat in stats.items():
        if len(stat['cpu_samples']) <= 0:
          continue
        avg_cpu = np.array(stat['cpu_samples']).mean()
        c = len(stat['cpu_samples'])
        stat['cpu_samples'] = []
        if not stat['avg_cpu']:
          stat['avg_cpu'] = avg_cpu
        else:
          stat['avg_cpu'] = (stat['avg_cpu'] * (c + i) + avg_cpu * c) / (c + i + c)
        if not stat['min'] or avg_cpu < stat['min']:
          stat['min'] = avg_cpu
        if not stat['max'] or avg_cpu > stat['max']:
          stat['max'] = avg_cpu
        msg = 'avg: {1:.2f}%, min: {2:.2f}%, max: {3:.2f}% {0}'.format(os.path.basename(k), stat['avg_cpu'], stat['min'], stat['max'])
        l.append((os.path.basename(k), avg_cpu, msg))
        avg_cpus.append(avg_cpu)
      l.sort(key= lambda x: -x[1])
      for x in l:
        print(x[2])
      print('avg sum: {0:.2f}%\n'.format(
        sum([stat['avg_cpu'] for k, stat in stats.items()])
      ))
