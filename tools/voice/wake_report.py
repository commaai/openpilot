#!/usr/bin/env python3
# summarize the voice server's WAKE_LOG: one row per reply with server and device timing, then counts and medians
import json
import statistics
import sys
import time


def main(path: str) -> None:
  ev = [json.loads(ln) for ln in open(path) if ln.strip()]
  device = {e['id']: e for e in ev if e.get('kind') == 'device'}
  rows = []
  print(f"{'time':8} {'gap':>5} {'llm':>5} {'tts':>5} {'rtt':>5} {'play':>5} {'total':>6}  query -> reply")
  for e in ev:
    if e.get('kind') != 'reply':
      continue
    lat, d = e['latency'], device.get(e['id'], {})
    total = d.get('last_word_to_speaker')
    rows.append((lat.get('gap'), lat.get('llm'), lat.get('tts_first'), d.get('rtt_ms'), d.get('play_start'), total))
    cells = [f'{v:5.2f}' if isinstance(v, float) else f'{v if v is not None else "-":>5}' for v in rows[-1][:5]]
    when = time.strftime('%H:%M:%S', time.localtime(e['time']))
    print(f"{when} {' '.join(cells)} {total if total is not None else '-':>6}  {e['query'][:40]!r} -> {e['reply'][:50]!r}")

  kinds = [e.get('kind') for e in ev]
  print('\n' + ', '.join(f'{k}: {kinds.count(k)}' for k in ('wake', 'reply', 'bare', 'device', 'near_miss', 'llm_error')))
  for name, i in (('gap', 0), ('llm', 1), ('last word to speaker', 5)):
    vals = [r[i] for r in rows if isinstance(r[i], (int, float))]
    if vals:
      print(f'{name}: median {statistics.median(vals):.2f} s, max {max(vals):.2f} s over {len(vals)}')

  # link health every 10 s of streamed audio
  link = [e for e in ev if e.get('kind') == 'link']
  for name in ('rtt_ms', 'net_lag_max', 'stale_max', 'asr_max'):
    vals = [e[name] for e in link if isinstance(e.get(name), (int, float))]
    if vals:
      print(f'link {name}: median {statistics.median(vals):g}, max {max(vals):g} over {len(vals)}')


if __name__ == '__main__':
  main(sys.argv[1] if len(sys.argv) > 1 else '/home/batman/.local/share/voiceserver/wakes.jsonl')
