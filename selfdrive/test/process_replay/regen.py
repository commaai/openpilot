#!/usr/bin/env python3
import os
import argparse
import time
import capnp

from typing import Union, Iterable, Optional, List, Any, Dict, Tuple

from openpilot.selfdrive.test.process_replay.process_replay import CONFIGS, FAKEDATA, ProcessConfig, replay_process, get_process_config, \
                                                                   check_openpilot_enabled, get_custom_params_from_lr
from openpilot.selfdrive.test.update_ci_routes import upload_route
from openpilot.tools.lib.route import Route
from openpilot.tools.lib.framereader import FrameReader
from openpilot.tools.lib.logreader import LogReader
from openpilot.tools.lib.helpers import save_log


def regen_segment(
  lr: Union[LogReader, List[capnp._DynamicStructReader]], frs: Optional[Dict[str, Any]] = None,
  processes: Iterable[ProcessConfig] = CONFIGS, disable_tqdm: bool = False
) -> List[capnp._DynamicStructReader]:
  all_msgs = sorted(lr, key=lambda m: m.logMonoTime)
  custom_params = get_custom_params_from_lr(all_msgs)

  print("Replayed processes:", [p.proc_name for p in processes])
  print("\n\n", "*"*30, "\n\n", sep="")

  output_logs = replay_process(processes, all_msgs, frs, return_all_logs=True, custom_params=custom_params, disable_progress=disable_tqdm)

  return output_logs


def setup_data_readers(
    route: str, sidx: int, use_route_meta: bool, needs_driver_cam: bool = True, needs_road_cam: bool = True
) -> Tuple[LogReader, Dict[str, Any]]:
  if use_route_meta:
    r = Route(route)
    lr = LogReader(r.log_paths()[sidx])
    frs = {}
    if needs_road_cam and len(r.camera_paths()) > sidx and r.camera_paths()[sidx] is not None:
      frs['roadCameraState'] = FrameReader(r.camera_paths()[sidx])
    if needs_road_cam and  len(r.ecamera_paths()) > sidx and r.ecamera_paths()[sidx] is not None:
      frs['wideRoadCameraState'] = FrameReader(r.ecamera_paths()[sidx])
    if needs_driver_cam and len(r.dcamera_paths()) > sidx and r.dcamera_paths()[sidx] is not None:
      frs['driverCameraState'] = FrameReader(r.dcamera_paths()[sidx])
  else:
    lr = LogReader(f"cd:/{route.replace('|', '/')}/{sidx}/rlog.bz2")
    frs = {}
    if needs_road_cam:
      frs['roadCameraState'] = FrameReader(f"cd:/{route.replace('|', '/')}/{sidx}/fcamera.hevc")
      if next((True for m in lr if m.which() == "wideRoadCameraState"), False):
        frs['wideRoadCameraState'] = FrameReader(f"cd:/{route.replace('|', '/')}/{sidx}/ecamera.hevc")
    if needs_driver_cam:
      frs['driverCameraState'] = FrameReader(f"cd:/{route.replace('|', '/')}/{sidx}/dcamera.hevc")

  return lr, frs


def regen_and_save(
  route: str, sidx: int, processes: Union[str, Iterable[str]] = "all", outdir: str = FAKEDATA,
  upload: bool = False, use_route_meta: bool = False, disable_tqdm: bool = False
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
  lr, frs = setup_data_readers(route, sidx, use_route_meta,
                               needs_driver_cam="driverCameraState" in all_vision_pubs,
                               needs_road_cam="roadCameraState" in all_vision_pubs or "wideRoadCameraState" in all_vision_pubs)
  output_logs = regen_segment(lr, frs, replayed_processes, disable_tqdm=disable_tqdm)

  log_dir = os.path.join(outdir, time.strftime("%Y-%m-%d--%H-%M-%S--0", time.gmtime()))
  rel_log_dir = os.path.relpath(log_dir)
  rpath = os.path.join(log_dir, "rlog.bz2")

  os.makedirs(log_dir)
  save_log(rpath, output_logs, compress=True)

  print("\n\n", "*"*30, "\n\n", sep="")
  print("New route:", rel_log_dir, "\n")

  if not check_openpilot_enabled(output_logs):
    raise Exception("Route did not engage for long enough")

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
  parser.add_argument("--whitelist-procs", type=comma_separated_list, default=all_procs,
                      help="Comma-separated whitelist of processes to regen (e.g. controlsd,radard)")
  parser.add_argument("--blacklist-procs", type=comma_separated_list, default=[],
                      help="Comma-separated blacklist of processes to regen (e.g. controlsd,radard)")
  parser.add_argument("route", type=str, help="The source route")
  parser.add_argument("seg", type=int, help="Segment in source route")
  args = parser.parse_args()

  blacklist_set = set(args.blacklist_procs)
  processes = [p for p in args.whitelist_procs if p not in blacklist_set]
  regen_and_save(args.route, args.seg, processes=processes, upload=args.upload, outdir=args.outdir)
