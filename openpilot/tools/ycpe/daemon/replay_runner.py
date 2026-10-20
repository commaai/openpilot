#!/usr/bin/env python3
"""Replays openpilot processes over a range of segments with the checkout it runs from, for daemon/replay.py.

It runs in the checkout's environment (PYTHONPATH points at that checkout), so it imports nothing from ycpe.
argv[1] is JSON: {procs, segs: [[n, rlog path], ...], out}. It writes <out>/<n>/rlog.zst per segment as soon as
the replay is past it: the segment's original initData (so times line up with the route) plus every replayed
message. Then <out>/diff.json compares replayed outputs to the originals. stdout carries one JSON event a line.
"""
import bisect
import json
import math
import os
import sys
import time
from collections import defaultdict

# services nothing replays from and that only cost memory
DROP = {"thumbnail", "qRoadEncodeIdx", "roadEncodeIdx", "wideRoadEncodeIdx", "driverEncodeIdx", "qRoadEncodeData", "logMessage",
        "errLogMessage", "androidLog", "procLog", "qcomGnss", "rawAudioData", "soundPressure", "microphone", "uploaderState"}
PAIRS = 3000      # message pairs per service compared, spread over the range
TOP = 40          # signals in the diff summary
LAG = 200         # external messages past a segment's end before writing it: outputs trail their inputs


def emit(event: str, **data) -> None:
  print(json.dumps({"event": event, **data}), flush=True)


def flatten(d, prefix: str, out: dict) -> None:
  """Numeric leaves of a message dict under store-style paths; lists of up to 16 numbers are split per index.
  Deprecated fields and timestamps of other messages are left out: they differ by construction."""
  for k, v in d.items():
    if "deprecated" in k.lower() or k.endswith("MonoTime"):
      continue
    path = f"{prefix}/{k}"
    if isinstance(v, dict):
      flatten(v, path, out)
    elif isinstance(v, bool):
      out[path] = float(v)
    elif isinstance(v, (int, float)):
      out[path] = float(v)
    elif isinstance(v, list) and 0 < len(v) <= 16 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
      for i, x in enumerate(v):
        out[f"{path}/{i}"] = float(x)


def diff(orig: list, new: list, t0: int, ignore: set[str]) -> dict:
  """Per numeric signal of each replayed service: how far the replay is from the original, matched by time."""
  by_svc_orig, by_svc_new = defaultdict(list), defaultdict(list)
  for m in orig:
    by_svc_orig[m.which()].append(m)
  for m in new:
    by_svc_new[m.which()].append(m)
  services, signals = {}, []
  for svc, news in by_svc_new.items():
    olds = by_svc_orig.get(svc, [])
    services[svc] = {"original": len(olds), "replayed": len(news)}
    if not olds or not hasattr(getattr(news[0], svc), "to_dict"):  # lists of messages (sendcan, pandaStates) aren't compared
      continue
    times = [m.logMonoTime for m in olds]
    stats: dict[str, list] = {}  # path -> [n, sum |d|, max |d|, n differing, first differing t, sum o, sum o^2]
    for m in news[::max(1, len(news) // PAIRS)]:
      j = bisect.bisect_left(times, m.logMonoTime)
      o = min(olds[max(0, j - 1):j + 1], key=lambda x: abs(x.logMonoTime - m.logMonoTime))
      a, b = {}, {}
      flatten(getattr(o, svc).to_dict(), f"/{svc}", a)
      flatten(getattr(m, svc).to_dict(), f"/{svc}", b)
      for path, x in a.items():
        y = b.get(path)
        if y is None or path in ignore or not (math.isfinite(x) and math.isfinite(y)):
          continue
        s = stats.setdefault(path, [0, 0.0, 0.0, 0, None, 0.0, 0.0])
        d = abs(x - y)
        s[0] += 1
        s[1] += d
        s[2] = max(s[2], d)
        s[5] += x
        s[6] += x * x
        if d > 1e-3 * max(1.0, abs(x)):
          s[3] += 1
          if s[4] is None:
            s[4] = (m.logMonoTime - t0) / 1e9
    for path, (n, total, worst, differ, first, so, so2) in stats.items():
      std = math.sqrt(max(0.0, so2 / n - (so / n) ** 2))
      mean = total / n
      signals.append({"path": path, "meanAbs": mean, "maxAbs": worst, "differ": differ / n, "first": first, "std": std,
                      "score": mean / (std + 1e-3)})
  diverged = sorted((s for s in signals if s["differ"]), key=lambda s: (-s["score"], -s["maxAbs"]))
  return {"services": services, "signals": diverged[:TOP], "compared": len(signals), "diverged": len(diverged)}


def main() -> None:
  args = json.loads(sys.argv[1])
  procs, segs, out = args["procs"], args["segs"], args["out"]
  started = time.monotonic()
  from openpilot.selfdrive.test.process_replay import process_replay as pr
  from openpilot.tools.lib.logreader import LogReader

  cfgs = [pr.get_process_config(p) for p in procs]
  pubs = {s for c in cfgs for s in c.pubs} - {s for c in cfgs for s in c.subs}

  # the inputs, and where each segment's external messages end (the replay walks them in time order)
  msgs, bounds, inits, ends = [], [], {}, []
  for i, (n, path) in enumerate(segs):
    emit("load", seg=n, i=i, count=len(segs))
    lr = [m for m in LogReader(path) if m.which() not in DROP]
    lr.sort(key=lambda m: m.logMonoTime)
    inits[n] = next((m for m in lr if m.which() == "initData"), None)
    bounds.append(next((m.logMonoTime for m in lr if m.which() != "initData"), 0))
    msgs.extend(lr)
    ends.append(sum(1 for m in lr if m.which() in pubs) + (ends[-1] if ends else 0))
  t0 = next((m.logMonoTime for m in inits.values() if m is not None), bounds[0])

  # replayed messages, filed by segment as they come out; a segment is written once the replay is past it
  produced: dict[int, list] = defaultdict(list)
  written: set[int] = set()
  totals = {"messages": 0}

  def write(i: int) -> None:
    n = segs[i][0]
    os.makedirs(os.path.join(out, str(n)), exist_ok=True)
    seg_msgs = ([inits[n]] if inits[n] is not None else []) + sorted(produced.pop(i, []), key=lambda m: m.logMonoTime)
    path = os.path.join(out, str(n), "rlog.zst")
    _save(path + ".tmp", seg_msgs)
    os.replace(path + ".tmp", path)
    written.add(i)
    emit("segment", seg=n, messages=len(seg_msgs))

  original_get = pr.ProcessContainer.get_output_msgs

  def get_output_msgs(self, start_time):
    got = original_get(self, start_time)
    for m in got:
      i = max(0, bisect.bisect_right(bounds, m.logMonoTime) - 1)
      while i in written and i < len(segs) - 1:  # a straggler: it goes with the next segment
        i += 1
      produced[i].append(m)
    totals["messages"] += len(got)
    return got
  pr.ProcessContainer.get_output_msgs = get_output_msgs

  class Progress:
    """Stands in for process_replay's tqdm over the external messages."""
    def __init__(self, total=0, **kwargs):
      self.n, self.total, self.sent = 0, total, 0.0

    def update(self, k=1):
      self.n += k
      seg = bisect.bisect_right(ends, self.n)
      for i in range(seg):
        if i not in written and self.n >= ends[i] + LAG:
          write(i)
      now = time.monotonic()
      if now - self.sent > 0.25:
        self.sent = now
        elapsed = now - started
        emit("progress", done=self.n, total=self.total, seg=segs[min(seg, len(segs) - 1)][0], elapsed=elapsed)

    def close(self):
      pass
  pr.tqdm = Progress

  emit("replay", procs=procs, messages=len(msgs))
  output = pr.replay_process(cfgs, msgs, disable_progress=False)
  for i in range(len(segs)):
    if i not in written:
      write(i)

  emit("diffing")
  ignore = {"/" + f.replace(".", "/") for c in cfgs for f in c.ignore}
  replayed = {m.which() for m in output}
  summary = diff([m for m in msgs if m.which() in replayed], output, t0, ignore)
  summary.update(procs=procs, segments=[n for n, _ in segs], seconds=time.monotonic() - started, outputs=len(output))
  with open(os.path.join(out, "diff.json"), "w") as f:
    json.dump(summary, f)
  emit("done", seconds=summary["seconds"], diverged=summary["diverged"])


def _save(path: str, msgs: list) -> None:
  import zstandard
  data = b"".join(m.as_builder().to_bytes() for m in msgs)
  with open(path, "wb") as f:
    f.write(zstandard.ZstdCompressor(level=3).compress(data))


if __name__ == "__main__":
  main()
