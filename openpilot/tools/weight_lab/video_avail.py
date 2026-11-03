#!/usr/bin/env python3
"""Print which segments of each route have rlog + both full-res videos."""
import sys
from openpilot.tools.lib.route import Route

for r in sys.argv[1:]:
  try:
    rt = Route(r)
    segs = [s.name.segment_num for s in rt.segments if s.camera_path and s.ecamera_path and s.log_path]
    print(f'{r} {len(rt.segments)} segs, full video: {segs}', flush=True)
  except Exception as e:
    print(f'{r} ERROR {type(e).__name__}', flush=True)
