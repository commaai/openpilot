#!/usr/bin/env python3
"""Find bookmarks (bookmarkButton / userBookmark) in a dongle's recent routes, from qlogs.

usage: bookmarks.py DONGLE DAYS
"""
import sys
import time
from multiprocessing import Pool
from openpilot.tools.lib.api import CommaApi
from openpilot.tools.lib.auth_config import get_token
from openpilot.tools.lib.logreader import LogReader


def scan(route):
  hits = []
  try:
    t0 = None
    for m in LogReader(f'{route}/q', sort_by_time=True):
      if t0 is None:
        t0 = m.logMonoTime
      if m.which() in ('bookmarkButton', 'userBookmark'):
        hits.append((m.logMonoTime - t0) / 1e9)
  except Exception as e:
    return route, None, type(e).__name__
  return route, hits, ''


if __name__ == '__main__':
  dongle, days = sys.argv[1], float(sys.argv[2])
  api = CommaApi(get_token())
  now = time.time()
  rs = api.get(f'v1/devices/{dongle}/routes_segments', params={'start': int((now - days * 86400) * 1000), 'end': int(now * 1000)})
  routes = sorted({r['fullname'] for r in rs})
  with Pool(16) as p:
    for route, hits, err in p.imap(scan, routes):
      if err:
        print(route, 'ERROR', err, flush=True)
      elif hits:
        # collapse presses within 5 s of each other
        merged = [h for i, h in enumerate(hits) if i == 0 or h - hits[i - 1] > 5]
        print(route, ' '.join(f'{h:.0f}' for h in merged), flush=True)
