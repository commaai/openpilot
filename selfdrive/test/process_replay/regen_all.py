#!/usr/bin/env python3
import argparse
import concurrent.futures
import os
from tqdm import tqdm

from selfdrive.test.process_replay.helpers import OpenpilotPrefix
from selfdrive.test.process_replay.regen import regen_and_save
from selfdrive.test.process_replay.test_processes import FAKEDATA, original_segments as segments

def regen_job(segment):
  with OpenpilotPrefix():
    route = segment[1].rsplit('--', 1)[0]
    sidx = int(segment[1].rsplit('--', 1)[1])
    relr = regen_and_save(route, sidx, upload=True, use_route_meta=False, outdir=os.path.join(FAKEDATA, segment[0]))
    relr = "|".join(relr.split("/")[::2])
    return f'  ("{segment[0]}", "{relr}"), '

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Generate new segments from old ones")
  parser.add_argument("-j", "--jobs", type=int, default=1)
  args = parser.parse_args()
  with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
    p = pool.map(regen_job, segments)
    msg = "COPY THIS INTO test_processes.py"
    for seg in tqdm(p, desc="Generating segments", total=len(segments)):
      msg += "\n" + str(seg)
    print()
    print()
    print(msg)
