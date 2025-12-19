#!/usr/bin/env python3
import os
import argparse
import time
import capnp

from typing import Any
from collections.abc import Iterable

from openpilot.selfdrive.test.process_replay.process_replay import CONFIGS, FAKEDATA, ProcessConfig, replay_process, get_process_config, \
                                                                   check_openpilot_enabled, check_most_messages_valid, get_custom_params_from_lr
from openpilot.selfdrive.test.update_ci_routes import upload_route
from openpilot.tools.lib.framereader import FrameReader
from openpilot.tools.lib.logreader import LogReader, LogIterable, save_log
from openpilot.tools.lib.openpilotci import get_url


def regen_segment(
  lr: LogIterable, frs: dict[str, Any] = None,
  processes: Iterable[ProcessConfig] = CONFIGS, disable_tqdm: bool = False
) -> list[capnp._DynamicStructReader]:
  all_msgs = sorted(lr, key=lambda m: m.logMonoTime)
  custom_params = get_custom_params_from_lr(all_msgs)

  print("Replayed processes:", [p.proc_name for p in processes])
  print("\n\n", "*"*30, "\n\n", sep="")

  output_logs = replay_process(processes, all_msgs, frs, return_all_logs=True, custom_params=custom_params, disable_progress=disable_tqdm)

  return output_logs


def setup_data_readers(
    route: str, sidx: int, needs_driver_cam: bool = True, needs_road_cam: bool = True, dummy_driver_cam: bool = False
) -> tuple[LogReader, dict[str, Any]]:
  lr = LogReader(f"{route}/{sidx}/r")
  frs = {}
  if needs_road_cam:
    frs['roadCameraState'] = FrameReader(get_url(route, str(sidx), "fcamera.hevc"))
    if next((True for m in lr if m.which() == "wideRoadCameraState"), False):
      frs['wideRoadCameraState'] = FrameReader(get_url(route, str(sidx), "ecamera.hevc"))
  if needs_driver_cam:
    if dummy_driver_cam:
      frs['driverCameraState'] = FrameReader(get_url(route, str(sidx), "fcamera.hevc")) # Use fcam as dummy
    else:
      device_type = next(str(msg.initData.deviceType) for msg in lr if msg.which() == "initData")
      assert device_type != "neo", "Driver camera not supported on neo segments. Use dummy dcamera."
      frs['driverCameraState'] = FrameReader(get_url(route, str(sidx), "dcamera.hevc"))

  return lr, frs


def regen_and_save(
  route: str, sidx: int, processes: str | Iterable[str] = "all", outdir: str = FAKEDATA,
  upload: bool = False, disable_tqdm: bool = False, dummy_driver_cam: bool = False
) -> str:
  if not isinstance(processes, str) and not hasattr(processes, "__iter__"):
    raise ValueError("whitelist_proc must be a string or iterable")

  if processes != "all":
    if isinstance(processes, str):
      raise ValueError(f"Invalid value for processes: {processes}")

    replayed_processes = []
    for d in processes:
      cfg = get_process_config(d)
      replayed_processes.append(cfg)
  else:
    replayed_processes = CONFIGS

  all_vision_pubs = {pub for cfg in replayed_processes for pub in cfg.vision_pubs}
  lr, frs = setup_data_readers(route, sidx,
                               needs_driver_cam="driverCameraState" in all_vision_pubs,
                               needs_road_cam="roadCameraState" in all_vision_pubs or "wideRoadCameraState" in all_vision_pubs,
                               dummy_driver_cam=dummy_driver_cam)
  output_logs = regen_segment(lr, frs, replayed_processes, disable_tqdm=disable_tqdm)

  log_dir = os.path.join(outdir, time.strftime("%Y-%m-%d--%H-%M-%S--0", time.gmtime()))
  rel_log_dir = os.path.relpath(log_dir)
  rpath = os.path.join(log_dir, "rlog.zst")

  os.makedirs(log_dir)
  save_log(rpath, output_logs, compress=True)

  print("\n\n", "*"*30, "\n\n", sep="")
  print("New route:", rel_log_dir, "\n")

  if not check_openpilot_enabled(output_logs):
    raise Exception("Route did not engage for long enough")
  if not check_most_messages_valid(output_logs):
    raise Exception("Route has too many invalid messages")

  if upload:
    upload_route(rel_log_dir)

  return rel_log_dir


if __name__ == "__main__":
  def comma_separated_list(string):
    return string.split(",")

  all_procs = [p.proc_name for p in CONFIGS]
  parser = argparse.ArgumentParser(description="Generate new segments from old ones")
  parser.add_argument("--upload", action="store_true", help="Upload the new segment to the CI bucket")
  parser.add_argument("--outdir", help="log output dir", default=FAKEDATA)
  parser.add_argument("--dummy-dcamera", action='store_true', help="Use dummy blank driver camera")
  parser.add_argument("--whitelist-procs", type=comma_separated_list, default=all_procs,
                      help="Comma-separated whitelist of processes to regen (e.g. controlsd,radard)")
  parser.add_argument("--blacklist-procs", type=comma_separated_list, default=[],
                      help="Comma-separated blacklist of processes to regen (e.g. controlsd,radard)")
  parser.add_argument("route", type=str, help="The source route")
  parser.add_argument("seg", type=int, help="Segment in source route")
  args = parser.parse_args()

  blacklist_set = set(args.blacklist_procs)
  processes = [p for p in args.whitelist_procs if p not in blacklist_set]
  regen_and_save(args.route, args.seg, processes=processes, upload=args.upload, outdir=args.outdir, dummy_driver_cam=args.dummy_dcamera)
