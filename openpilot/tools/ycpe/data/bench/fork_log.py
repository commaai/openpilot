#!/usr/bin/env python3
"""A fork's log for testing log schemas, without a fork route: a segment re-encoded with sunnypilot's schema, initData
pointing at a sunnypilot commit, plus sunnypilot-only events (selfdriveStateSP, carStateSP) beside every carState.
usage: fork_log.py <rlog.zst in> <rlog.zst out> <schema dir from capnpc.mjs --keep> <commit>"""
import sys
from pathlib import Path

import capnp
import zstandard

if __name__ == "__main__":
  src, dst, schema, commit = sys.argv[1:5]
  capnp.remove_import_hook()
  log = capnp.load(str(Path(schema) / "cereal" / "log.capnp"), imports=[str(Path(schema) / "imp")])
  out = []
  for ev in log.Event.read_multiple_bytes(zstandard.decompress(Path(src).read_bytes(), max_output_size=1 << 30)):
    b = ev.as_builder()
    if b.which() == "initData":
      b.initData.gitRemote = "https://github.com/sunnypilot/sunnypilot.git"
      b.initData.gitCommit = commit
      b.initData.gitBranch = "master"
    out.append(b.to_bytes())
    if b.which() == "carState":
      v = b.carState.vEgo
      sp = log.Event.new_message(logMonoTime=b.logMonoTime, valid=True)
      mads = sp.init("selfdriveStateSP").init("mads")
      mads.state, mads.enabled, mads.active, mads.available = ("enabled" if v > 5 else "paused"), v > 5, v > 5, True
      sp.selfdriveStateSP.buttonsPressed = int(v) % 7
      out.append(sp.to_bytes())
      out.append(log.Event.new_message(logMonoTime=b.logMonoTime, valid=True, carStateSP={"speedLimit": round(v * 1.1, 1)}).to_bytes())
  Path(dst).write_bytes(zstandard.compress(b"".join(out), 3))
  print(f"{dst}: {len(out)} events")
