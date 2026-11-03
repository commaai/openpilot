#!/usr/bin/env python3
"""Count radar-tracked lead frames per segment, to pick segments for the lead study."""
import sys
from multiprocessing import Pool
from openpilot.tools.lib.logreader import LogReader


def count(seg):
  n = 0
  dv = 0
  for m in LogReader(seg, sort_by_time=True):
    if m.which() == 'radarState':
      lead = m.radarState.leadOne
      if lead.present and lead.radar:
        n += 1
        dv += abs(lead.vRel) > 3
  return seg, n, dv


if __name__ == '__main__':
  route, nseg = sys.argv[1], int(sys.argv[2])
  with Pool(8) as p:
    for seg, n, dv in p.imap(count, [f'{route}/{i}' for i in range(nseg)]):
      print(seg.rsplit('/', 1)[1], n, dv, flush=True)
