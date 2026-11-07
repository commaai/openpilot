#!/usr/bin/env python3
# runs on a mac in the car: plays labeled wake word and negative clips while the comma four records audio
# usage: python3 play_car.py <kit dir> [--minutes 40]
import argparse, json, os, random, subprocess, time

SYNC_EVERY_S = 120


def play(path: str, volume: float) -> None:
  subprocess.run(['afplay', '-v', f'{volume:.2f}', path], check=True)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument('kit')
  ap.add_argument('--minutes', type=float, default=40)
  ap.add_argument('--min-gap', type=float, default=15)
  ap.add_argument('--max-gap', type=float, default=40)
  args = ap.parse_args()

  clips = json.load(open(os.path.join(args.kit, 'manifest.json')))
  sync = os.path.join(args.kit, 'sync.wav')
  log_path = os.path.join(args.kit, f'log_{time.strftime("%Y%m%d_%H%M%S")}.jsonl')
  end = time.time() + args.minutes * 60
  last_sync = 0.0
  with open(log_path, 'w') as log:
    while time.time() < end:
      if time.time() - last_sync > SYNC_EVERY_S:
        t = time.time(); play(sync, 0.6); last_sync = t
        log.write(json.dumps({'t': t, 'sync': True}) + '\n'); log.flush()
      c = random.choice(clips)
      vol = random.choice((0.25, 0.4, 0.6, 0.8))
      t = time.time(); play(os.path.join(args.kit, c['file']), vol)
      log.write(json.dumps({'t': t, 'file': c['file'], 'y': c['y'], 'text': c['text'], 'vol': vol}) + '\n'); log.flush()
      print(f"{time.strftime('%H:%M:%S')} {'WAKE' if c['y'] else 'neg '} vol {vol} {c['text']}")
      time.sleep(random.uniform(args.min_gap, args.max_gap))
  print('log written to', log_path)


if __name__ == '__main__':
  main()
