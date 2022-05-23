#!/usr/bin/env python3
import argparse
import concurrent.futures

from selfdrive.test.process_replay.regen import regen_and_save
from selfdrive.test.process_replay.test_processes import original_segments as segments

def regen_job(segment):
  route = segment[1].rsplit('--', 1)[0]
  sidx = int(segment[1].rsplit('--', 1)[1])
  msg = "Regen  " + route + "  " + sidx + "\n"
  relr = regen_and_save(route, sidx, upload=True, use_route_meta=False)
  msg += "\n\n  " + "*"*30 + "  \n\n" + "New route:  " + relr
  print(msg)
  relr = relr.replace('/', '|')
  return f'  ("{segment[0]}", "{relr}"), '

if __name__ == "__main__":
  #TODO description
  parser = argparse.ArgumentParser(description="")
  parser.add_argument("-j", "--jobs", type=int, default=1)
  args = parser.parse_args()
  with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
    p1 = pool.map(regen_job, segments)
    print()
    print()
    print()
    print('COPY THIS INTO test_processes.py')
    for seg in p1:
      print(seg)
