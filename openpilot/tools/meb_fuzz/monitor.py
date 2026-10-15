# Reads a stream of status.json dumps (one per line) from the device and prints a line whenever the state changes.
import json
import sys

last = None
for line in sys.stdin:
  line = line.strip()
  if not line.startswith('{'):
    continue
  try:
    s = json.loads(line)
  except json.JSONDecodeError:
    continue
  g = s['signals']
  run = s['run'] or {}
  key = (s['alert'][0], g['hms'], g['ss'], g['motion'], g['epb'], g['tsk'], g['cam'], g['req'], g['long_active'], g['gas'],
         g['brake'], g['acc_faulted'], g['can_valid'], g['gear'], run.get('phase'), run.get('step'), run.get('outcome'))
  if key != last:
    print(f"{s['time'][11:23]} v={g['v']:5.2f} req={g['req']} hms={g['hms']} ss={g['ss']} mot={g['motion']} epb={g['epb']} "
          f"tsk={g['tsk']} cam={g['cam']} la={int(g['long_active'])} gas={int(g['gas'])} brk={int(g['brake'])} "
          f"flt={int(g['acc_faulted'])} gear={g['gear']} | {s['alert'][0]} | {s['alert'][1]}", flush=True)
    last = key
