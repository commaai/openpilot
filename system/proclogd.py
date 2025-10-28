#!/usr/bin/env python3
import os
from typing import NoReturn, TypedDict

from cereal import messaging
from openpilot.common.realtime import Ratekeeper
from openpilot.common.swaglog import cloudlog

JIFFY = os.sysconf(os.sysconf_names['SC_CLK_TCK'])
PAGE_SIZE = os.sysconf(os.sysconf_names['SC_PAGE_SIZE'])


def _cpu_times() -> list[dict[str, float]]:
  cpu_times: list[dict[str, float]] = []
  try:
    with open('/proc/stat') as f:
      lines = f.readlines()[1:]
    for line in lines:
      if not line.startswith('cpu') or len(line) < 4 or not line[3].isdigit():
        break
      parts = line.split()
      cpu_times.append({
        'cpuNum': int(parts[0][3:]),
        'user': float(parts[1]) / JIFFY,
        'nice': float(parts[2]) / JIFFY,
        'system': float(parts[3]) / JIFFY,
        'idle': float(parts[4]) / JIFFY,
        'iowait': float(parts[5]) / JIFFY,
        'irq': float(parts[6]) / JIFFY,
        'softirq': float(parts[7]) / JIFFY,
      })
  except Exception:
    cloudlog.exception("failed to read /proc/stat")
  return cpu_times


def _mem_info() -> dict[str, int]:
  keys = ["MemTotal:", "MemFree:", "MemAvailable:", "Buffers:", "Cached:", "Active:", "Inactive:", "Shmem:"]
  info: dict[str, int] = dict.fromkeys(keys, 0)
  try:
    with open('/proc/meminfo') as f:
      for line in f:
        parts = line.split()
        if parts and parts[0] in info:
          info[parts[0]] = int(parts[1]) * 1024
  except Exception:
    cloudlog.exception("failed to read /proc/meminfo")
  return info


_STAT_POS = {
  'pid': 1,
  'state': 3,
  'ppid': 4,
  'utime': 14,
  'stime': 15,
  'cutime': 16,
  'cstime': 17,
  'priority': 18,
  'nice': 19,
  'num_threads': 20,
  'starttime': 22,
  'vsize': 23,
  'rss': 24,
  'processor': 39,
}

class ProcStat(TypedDict):
  name: str
  pid: int
  state: str
  ppid: int
  utime: int
  stime: int
  cutime: int
  cstime: int
  priority: int
  nice: int
  num_threads: int
  starttime: int
  vms: int
  rss: int
  processor: int


def _parse_proc_stat(stat: str) -> ProcStat | None:
  open_paren = stat.find('(')
  close_paren = stat.rfind(')')
  if open_paren == -1 or close_paren == -1 or open_paren > close_paren:
    return None
  name = stat[open_paren + 1:close_paren]
  stat = stat[:open_paren] + stat[open_paren:close_paren].replace(' ', '_') + stat[close_paren:]
  parts = stat.split()
  if len(parts) < 52:
    return None
  try:
    return {
      'name': name,
      'pid': int(parts[_STAT_POS['pid'] - 1]),
      'state': parts[_STAT_POS['state'] - 1][0],
      'ppid': int(parts[_STAT_POS['ppid'] - 1]),
      'utime': int(parts[_STAT_POS['utime'] - 1]),
      'stime': int(parts[_STAT_POS['stime'] - 1]),
      'cutime': int(parts[_STAT_POS['cutime'] - 1]),
      'cstime': int(parts[_STAT_POS['cstime'] - 1]),
      'priority': int(parts[_STAT_POS['priority'] - 1]),
      'nice': int(parts[_STAT_POS['nice'] - 1]),
      'num_threads': int(parts[_STAT_POS['num_threads'] - 1]),
      'starttime': int(parts[_STAT_POS['starttime'] - 1]),
      'vms': int(parts[_STAT_POS['vsize'] - 1]),
      'rss': int(parts[_STAT_POS['rss'] - 1]),
      'processor': int(parts[_STAT_POS['processor'] - 1]),
    }
  except Exception:
    cloudlog.exception("failed to parse /proc/<pid>/stat")
    return None

class ProcExtra(TypedDict):
  pid: int
  name: str
  exe: str
  cmdline: list[str]


_proc_cache: dict[int, ProcExtra] = {}


def _get_proc_extra(pid: int, name: str) -> ProcExtra:
  cache: ProcExtra | None = _proc_cache.get(pid)
  if cache is None or cache.get('name') != name:
    exe = ''
    cmdline: list[str] = []
    try:
      exe = os.readlink(f'/proc/{pid}/exe')
    except OSError:
      pass
    try:
      with open(f'/proc/{pid}/cmdline', 'rb') as f:
        cmdline = [c.decode('utf-8', errors='replace') for c in f.read().split(b'\0') if c]
    except OSError:
      pass
    cache = {'pid': pid, 'name': name, 'exe': exe, 'cmdline': cmdline}
    _proc_cache[pid] = cache
  return cache


def _procs() -> list[ProcStat]:
  stats: list[ProcStat] = []
  for pid_str in os.listdir('/proc'):
    if not pid_str.isdigit():
      continue
    try:
      with open(f'/proc/{pid_str}/stat') as f:
        stat = f.read()
      parsed = _parse_proc_stat(stat)
      if parsed is not None:
        stats.append(parsed)
    except OSError:
      continue
  return stats


def build_proc_log_message(msg) -> None:
  pl = msg.procLog

  procs = _procs()
  l = pl.init('procs', len(procs))
  for i, r in enumerate(procs):
    proc = l[i]
    proc.pid = r['pid']
    proc.state = ord(r['state'][0])
    proc.ppid = r['ppid']
    proc.cpuUser = r['utime'] / JIFFY
    proc.cpuSystem = r['stime'] / JIFFY
    proc.cpuChildrenUser = r['cutime'] / JIFFY
    proc.cpuChildrenSystem = r['cstime'] / JIFFY
    proc.priority = r['priority']
    proc.nice = r['nice']
    proc.numThreads = r['num_threads']
    proc.startTime = r['starttime'] / JIFFY
    proc.memVms = r['vms']
    proc.memRss = r['rss'] * PAGE_SIZE
    proc.processor = r['processor']
    proc.name = r['name']

    extra = _get_proc_extra(r['pid'], r['name'])
    proc.exe = extra['exe']
    cmdline = proc.init('cmdline', len(extra['cmdline']))
    for j, arg in enumerate(extra['cmdline']):
      cmdline[j] = arg

  cpu_times = _cpu_times()
  cpu_list = pl.init('cpuTimes', len(cpu_times))
  for i, ct in enumerate(cpu_times):
    cpu = cpu_list[i]
    cpu.cpuNum = ct['cpuNum']
    cpu.user = ct['user']
    cpu.nice = ct['nice']
    cpu.system = ct['system']
    cpu.idle = ct['idle']
    cpu.iowait = ct['iowait']
    cpu.irq = ct['irq']
    cpu.softirq = ct['softirq']

  mem_info = _mem_info()
  pl.mem.total = mem_info["MemTotal:"]
  pl.mem.free = mem_info["MemFree:"]
  pl.mem.available = mem_info["MemAvailable:"]
  pl.mem.buffers = mem_info["Buffers:"]
  pl.mem.cached = mem_info["Cached:"]
  pl.mem.active = mem_info["Active:"]
  pl.mem.inactive = mem_info["Inactive:"]
  pl.mem.shared = mem_info["Shmem:"]


def main() -> NoReturn:
  pm = messaging.PubMaster(['procLog'])
  rk = Ratekeeper(0.5)
  while True:
    msg = messaging.new_message('procLog', valid=True)
    build_proc_log_message(msg)
    pm.send('procLog', msg)
    rk.keep_time()


if __name__ == '__main__':
  main()
