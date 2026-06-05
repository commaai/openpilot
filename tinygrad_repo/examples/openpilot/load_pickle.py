import sys, pickle
from extra.bench_log import WallTimeEvent, BenchEvent
from tinygrad.helpers import getenv

PKL = sys.argv[1] if len(sys.argv) > 1 else "/tmp/openpilot.pkl"

load_times = []

for _ in range(10):
  with WallTimeEvent(BenchEvent.STEP) as wte: pickle.load(open(PKL, 'rb'))
  load_times.append(wte.time)
  print(f"pickle load: {wte.time:6.2f} s")

if (assert_time:=getenv("ASSERT_MIN_LOAD_TIME")):
  min_time = min(load_times)
  assert min_time < assert_time, f"Speed regression, expected min load time of < {assert_time} s but took: {min_time} s"
