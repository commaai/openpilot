# One compact line per finished run from a stream of results.jsonl lines.
import json
import sys

for line in sys.stdin:
  line = line.strip()
  if not line.startswith('{'):
    if line:
      print(f'NOTE {line}', flush=True)
    continue
  try:
    r = json.loads(line)
  except json.JSONDecodeError:
    continue
  ev = r['events']
  seq = {}
  for t, k, v in ev:
    if k in ('hms', 'req', 'tsk', 'cam', 'ss', 'motion', 'state'):
      seq.setdefault(k, [])
      if not seq[k] or seq[k][-1][1] != v:
        seq[k].append((t, v))
  fmt = lambda k: ' '.join(f'{v}@{t:.2f}' for t, v in seq.get(k, []))
  notes = [v for t, k, v in ev if k == 'note']
  print(f"RUN #{r['id']} {r['program'].get('name', '')} {r['variants']} -> {r['outcome']} "
        f"(dur {r['duration']}s, max_v {r['max_v']}) | req {fmt('req')} | hms {fmt('hms')} | ss {fmt('ss')} | "
        f"tsk {fmt('tsk')} | cam {fmt('cam')} | notes {notes[-4:]}", flush=True)
