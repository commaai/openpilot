#!/usr/bin/env python3
# Event-level telemetry for the big/small model handover.
import json
import os
import time

import openpilot.cereal.messaging as messaging
from openpilot.common.params import Params

OUT = "/data/fallback_watch.jsonl"

def proc_info(name):
  out = {}
  for pid in os.listdir("/proc"):
    if not pid.isdigit():
      continue
    try:
      with open(f"/proc/{pid}/cmdline") as f:
        cmd = f.read()
      if name not in cmd:
        continue
      with open(f"/proc/{pid}/stat") as f:
        stat = f.read().rsplit(")", 1)[1].split()
      with open(f"/proc/{pid}/status") as f:
        status = f.read()
      rss_kb = 0
      for line in status.splitlines():
        if line.startswith("VmRSS:"):
          rss_kb = int(line.split()[1])
      # fields after comm: state=idx0, so psr=idx36, rt_priority=idx37, policy=idx38
      psr = int(stat[36])
      rtprio = int(stat[37]) if len(stat) > 37 else -1
      policy = int(stat[38]) if len(stat) > 38 else -1
      out = {"pid": int(pid), "core": psr, "rtprio": rtprio, "policy": policy, "rss_mb": rss_kb // 1024}
      break
    except Exception:
      pass
  return out

def main():
  params = Params()
  sm = messaging.SubMaster(["modelV2", "selfdriveState", "carControl"])
  last_msg_t = None
  last_frame = -1
  last_summary_t = time.monotonic()
  phase = "big" if params.get_bool("UsbGpuActive") else "small_loading"
  phase_start_t = time.monotonic()
  gaps = []

  def emit(event, **values):
    rec = {"mono": time.monotonic(), "event": event, "phase": phase, **values}
    with open(OUT, "a") as f:
      f.write(json.dumps(rec) + "\n")

  emit("start")
  while True:
    sm.update(20)
    now = time.monotonic()
    active = params.get_bool("UsbGpuActive")
    new_phase = "big" if active else ("small_fallback" if phase == "big" else phase)
    if new_phase != phase:
      old_phase = phase
      phase = new_phase
      emit("phase", old=old_phase, new=phase, elapsed=round(now - phase_start_t, 3))
      phase_start_t = now
      gaps.clear()

    if sm.updated["modelV2"] and sm["modelV2"].frameId != last_frame:
      # gap between publish times, immune to this script stalling
      pub_t = sm.logMonoTime["modelV2"] / 1e9
      if last_msg_t is not None:
        gap_ms = (pub_t - last_msg_t) * 1000
        gaps.append(gap_ms)
        if gap_ms > 75:
          emit("lag", gap_ms=round(gap_ms, 2), frame_id=sm["modelV2"].frameId,
               execution_ms=round(sm["modelV2"].modelExecutionTime * 1000, 2))
      last_msg_t = pub_t
      last_frame = sm["modelV2"].frameId

    if now - last_summary_t >= 1:
      ordered = sorted(gaps)
      p99 = ordered[min(len(ordered) - 1, int(len(ordered) * .99))] if ordered else 0
      emit("summary", active=active, enabled=sm["carControl"].enabled, state=str(sm["selfdriveState"].state),
           messages=len(gaps), max_gap_ms=round(max(gaps), 2) if gaps else 0,
           p99_gap_ms=round(p99, 2), age_ms=round((now - last_msg_t) * 1000, 2) if last_msg_t else -1,
           frame_drop_perc=round(sm["modelV2"].frameDropPerc, 2),
           execution_ms=round(sm["modelV2"].modelExecutionTime * 1000, 2), modeld=proc_info("modeld.modeld"))
      last_summary_t = now

if __name__ == "__main__":
  main()
