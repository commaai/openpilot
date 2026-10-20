#!/usr/bin/env python3
"""Replays openpilot processes over a range of segments with the checkout it runs from, for daemon/replay.py.

It runs in the checkout's environment (PYTHONPATH points at that checkout), so it imports nothing from ycpe.
argv[1] is JSON: {procs, segs: [[n, rlog path], ...], out, warm, videos: {n: {camera state: hevc path}}}. It writes <out>/<n>/rlog.zst per segment as soon
as the replay is past it: the segment's original initData (so times line up with the route) plus every replayed
message. Then <out>/diff.json compares replayed outputs to the originals. stdout carries one JSON event a line.
With warm, the first segment only warms the processes up (engagement, calibration, learners): nothing of it is
written or compared. Vision processes (modeld, dmonitoringmodeld) are fed the segments' video frames, as in
process_replay's model replay.
"""
import bisect
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict, deque

# services nothing replays from and that only cost memory
DROP = {"thumbnail", "qRoadEncodeIdx", "roadEncodeIdx", "wideRoadEncodeIdx", "driverEncodeIdx", "qRoadEncodeData", "logMessage",
        "errLogMessage", "androidLog", "procLog", "qcomGnss", "rawAudioData", "soundPressure", "microphone", "uploaderState"}
PAIRS = 3000      # message pairs per service compared, spread over the range
TOP = 40          # signals in the diff summary
LAG = int(2e9)    # ns of outputs past a segment's end before writing it: outputs trail their inputs
ENCODE_IDX = {"narrowRoadEncodeIdx": "narrowRoadCameraState", "wideRoadEncodeIdx": "wideRoadCameraState", "cabinEncodeIdx": "cabinCameraState"}


def emit(event: str, **data) -> None:
  print(json.dumps({"event": event, **data}), flush=True)


def flatten(d, prefix: str, out: dict) -> None:
  """Numeric leaves of a message dict under store-style paths; lists of up to 16 numbers are split per index.
  Deprecated fields, timestamps of other messages and frame ids (replayed camera states count frames within their
  segment) are left out: they differ by construction."""
  for k, v in d.items():
    if "deprecated" in k.lower() or k.endswith("MonoTime") or k.startswith(("timestamp", "frameId")):
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
  procs, segs, out, warm = args["procs"], args["segs"], args["out"], bool(args.get("warm"))
  started = time.monotonic()
  from openpilot.selfdrive.test.process_replay import process_replay as pr
  from openpilot.tools.lib.logreader import LogReader

  cfgs = [pr.get_process_config(p) for p in procs]

  # the inputs, and where each segment starts: its most frequent service's first message (a segment's log can hold
  # a few messages from before its start)
  msgs, bounds, inits = [], [], {}
  for i, (n, path) in enumerate(segs):
    emit("load", seg=n, i=i, count=len(segs))
    lr = [m for m in LogReader(path) if m.which() not in DROP]
    lr.sort(key=lambda m: m.logMonoTime)
    inits[n] = next((m for m in lr if m.which() == "initData"), None)
    common = Counter(m.which() for m in lr if m.which() != "initData").most_common(1)
    bounds.append(next((m.logMonoTime for m in lr if common and m.which() == common[0][0]), 0))
    msgs.extend(lr)
  t0 = next((m.logMonoTime for m in inits.values() if m is not None), bounds[0])
  start = bounds[1] if warm and len(segs) > 1 else 0  # outputs before it are the warm-up's

  # replayed messages, filed by segment by time; a segment is written once outputs are LAG past its end
  produced: dict[int, list] = defaultdict(list)
  written: set[int] = set()
  latest = {"t": 0, "seg": 0}

  def write(i: int) -> None:
    n = segs[i][0]
    written.add(i)
    if warm and i == 0:
      emit("warmed", seg=n)
      return
    os.makedirs(os.path.join(out, str(n)), exist_ok=True)
    seg_msgs = ([inits[n]] if inits[n] is not None else []) + sorted(produced.pop(i, []), key=lambda m: m.logMonoTime)
    path = os.path.join(out, str(n), "rlog.zst")
    _save(path + ".tmp", seg_msgs)
    os.replace(path + ".tmp", path)
    emit("segment", seg=n, messages=len(seg_msgs))

  original_get = pr.ProcessContainer.get_output_msgs

  def get_output_msgs(self, start_time):
    got = original_get(self, start_time)
    for m in got:
      latest["t"] = max(latest["t"], m.logMonoTime)
      if m.logMonoTime < start:
        continue
      i = max(0, bisect.bisect_right(bounds, m.logMonoTime) - 1)
      while i in written and i < len(segs) - 1:  # a straggler past LAG: it goes with the next segment
        i += 1
      produced[i].append(m)
    latest["seg"] = max(0, bisect.bisect_right(bounds, latest["t"]) - 1)
    for i in range(latest["seg"]):
      if i not in written and latest["t"] > bounds[i + 1] + LAG:
        write(i)
    return got
  pr.ProcessContainer.get_output_msgs = get_output_msgs

  class Progress:
    """Stands in for process_replay's tqdm over the external messages."""
    def __init__(self, total=0, **kwargs):
      self.n, self.total, self.sent = 0, total, 0.0

    def update(self, k=1):
      self.n += k
      now = time.monotonic()
      if now - self.sent > 0.25:
        self.sent = now
        emit("progress", done=self.n, total=self.total, seg=segs[latest["seg"]][0], warm=warm and latest["seg"] == 0, elapsed=now - started)

    def close(self):
      pass
  pr.tqdm = Progress

  # learned state (calibration, vehicle and torque params) starts where the log had it, not from scratch
  try:
    custom_params = pr.get_custom_params_from_lr(msgs, "first")
  except (AttributeError, AssertionError):
    custom_params = None
  frs = videos(pr, cfgs, msgs, bounds, segs, {int(n): v for n, v in args.get("videos", {}).items()})
  emit("replay", procs=procs, messages=len(msgs))
  output = pr.replay_process(cfgs, msgs, frs=frs, custom_params=custom_params, disable_progress=False)
  for i in range(len(segs)):
    if i not in written:
      write(i)

  emit("diffing")
  ignore = {"/" + f.replace(".", "/") for c in cfgs for f in c.ignore}
  output = [m for m in output if m.logMonoTime >= start]
  replayed = {m.which() for m in output}
  summary = diff([m for m in msgs if m.which() in replayed and m.logMonoTime >= start], output, t0, ignore)
  summary.update(procs=procs, segments=[n for n, _ in segs[int(warm):]], warm=segs[0][0] if warm else None,
                 seconds=time.monotonic() - started, outputs=len(output))
  with open(os.path.join(out, "diff.json"), "w") as f:
    json.dump(summary, f)
  emit("done", seconds=summary["seconds"], diverged=summary["diverged"])


class Frames:
  """One camera's frames over several segments, for process_replay's frs. Its camera states are renumbered to frame
  indexes within their segment's video, so each get() takes the segment from the queue run_step filled."""
  def __init__(self, paths: dict[int, str]):
    from openpilot.tools.lib.framereader import FrameReader
    self.reader = lambda n: FrameReader(paths[n], pix_fmt="nv12", cache_size=20)
    self.paths, self.queue, self.open = paths, deque(), {}
    first = self.reader(min(paths))
    self.w, self.h, self.pix_fmt = first.w, first.h, first.pix_fmt

  def get(self, fid: int):
    n = self.queue.popleft()
    if n not in self.open:
      self.open = {n: self.reader(n)}  # segments go by in order: drop the last one's
    return self.open[n].get(fid)


def videos(pr, cfgs, msgs, bounds, segs, paths: dict[int, dict[str, str]]) -> dict | None:
  """Frame sources for the vision processes, and a run_step that tells them which segment each camera state is from."""
  cameras = {c for cfg in cfgs for c in cfg.vision_pubs}
  if not cameras:
    return None
  # each camera state's video segment, from the encoder's index (logged with the original frame ids), else by time
  encoded = {(ENCODE_IDX[m.which()], getattr(m, m.which()).frameId): getattr(m, m.which()).segmentNum for m in msgs if m.which() in ENCODE_IDX}
  where = {}
  for m in msgs:
    if m.which() in cameras:
      n = encoded.get((m.which(), getattr(m, m.which()).frameId))
      where[(m.which(), m.logMonoTime)] = n if n is not None else segs[max(0, bisect.bisect_right(bounds, m.logMonoTime) - 1)][0]
  frs = {c: Frames({n: v[c] for n, v in paths.items() if c in v}) for c in cameras if any(c in v for v in paths.values())}

  original_run_step = pr.ProcessContainer.run_step
  def run_step(self, msg, frs):
    if self.vipc_server is not None and msg.which() in self.cfg.vision_pubs:  # it sends this one's frame at the end of the cycle
      frs[msg.which()].queue.append(where[(msg.which(), msg.logMonoTime)])
    return original_run_step(self, msg, frs)
  pr.ProcessContainer.run_step = run_step
  return frs


def _save(path: str, msgs: list) -> None:
  import zstandard
  data = b"".join(m.as_builder().to_bytes() for m in msgs)
  with open(path, "wb") as f:
    f.write(zstandard.ZstdCompressor(level=3).compress(data))


if __name__ == "__main__":
  main()
