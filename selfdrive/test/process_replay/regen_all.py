#!/usr/bin/env python3
import argparse
import concurrent.futures
import os
import random
import traceback
from tqdm import tqdm

from openpilot.common.prefix import OpenpilotPrefix
from openpilot.selfdrive.test.process_replay.regen import regen_and_save
from openpilot.selfdrive.test.process_replay.test_processes import FAKEDATA, source_segments as segments
from openpilot.tools.lib.route import SegmentName


def regen_job(segment, upload, disable_tqdm):
  with OpenpilotPrefix():
    sn = SegmentName(segment[1])
    fake_dongle_id = 'regen' + ''.join(random.choice('0123456789ABCDEF') for _ in range(11))
    try:
      relr = regen_and_save(sn.route_name.canonical_name, sn.segment_num, upload=upload,
                            outdir=os.path.join(FAKEDATA, fake_dongle_id), disable_tqdm=disable_tqdm, dummy_driver_cam=True)
      relr = '|'.join(relr.split('/')[-2:])
      return f'  ("{segment[0]}", "{relr}"), '
    except Exception as e:
      err = f"  {segment} failed: {str(e)}"
      err += traceback.format_exc()
      err += "\n\n"
      return err


if __name__ == "__main__":
  all_cars = {car for car, _ in segments}

  parser = argparse.ArgumentParser(description="Generate new segments from old ones")
  parser.add_argument("-j", "--jobs", type=int, default=1)
  parser.add_argument("--no-upload", action="store_true")
  parser.add_argument("--whitelist-cars", type=str, nargs="*", default=all_cars,
                      help="Whitelist given cars from the test (e.g. HONDA)")
  parser.add_argument("--blacklist-cars", type=str, nargs="*", default=[],
                      help="Blacklist given cars from the test (e.g. HONDA)")
  args = parser.parse_args()

  tested_cars = set(args.whitelist_cars) - set(args.blacklist_cars)
  tested_cars = {c.upper() for c in tested_cars}
  tested_segments = [(car, segment) for car, segment in segments if car in tested_cars]

  with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
    p = pool.map(regen_job, tested_segments, [not args.no_upload] * len(tested_segments), [args.jobs > 1] * len(tested_segments))
    msg = "Copy these new segments into test_processes.py:"
    for seg in tqdm(p, desc="Generating segments", total=len(tested_segments)):
      msg += "\n" + str(seg)
    print()
    print()
    print(msg)
