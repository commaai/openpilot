#!/usr/bin/env python3

from selfdrive.test.openpilotci import get_url
from selfdrive.test.process_replay.regen import regen_segment
from selfdrive.test.process_replay.test_processes import segments
from tools.lib.framereader import FrameReader
from tools.lib.logreader import LogReader

def main():
  segs = dict.fromkeys(segments)
  for s in segs:
    r, n = s[1].rsplit("--", 1)
    lr = LogReader(get_url(r, n))
    fr = FrameReader(get_url(r, n, "fcamera"))
    segs[s] = regen_segment(lr, {'roadCameraState': fr})
  
  print("\n\n")
  print("Regenerated segments:")
  for old, new in segs.items():
    print(f"{old} -> {new}/0")

if __name__ == "__main__":
  main()
