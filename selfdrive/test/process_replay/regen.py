#!/usr/bin/env python3
import os
import argparse
import time
import capnp

from typing import Union, Iterable, Optional, List, Any, Dict, Tuple

from selfdrive.test.process_replay.process_replay import CONFIGS, FAKEDATA, replay_process, get_process_config, \
                                                        check_openpilot_enabled, get_custom_params_from_lr
from selfdrive.test.update_ci_routes import upload_route
from tools.lib.route import Route
from tools.lib.framereader import FrameReader
from tools.lib.logreader import LogReader
from tools.lib.helpers import save_log


def regen_segment(
  lr: Union[LogReader, List[capnp._DynamicStructReader]], frs: Optional[Dict[str, Any]] = None,
  daemons: Union[str, Iterable[str]] = "all", disable_tqdm: bool = False
) -> List[capnp._DynamicStructReader]:
  if not isinstance(daemons, str) and not hasattr(daemons, "__iter__"):
    raise ValueError("whitelist_proc must be a string or iterable")

  all_msgs = sorted(lr, key=lambda m: m.logMonoTime)
  custom_params = get_custom_params_from_lr(all_msgs)

  if daemons != "all":
    if isinstance(daemons, str):
      raise ValueError(f"Invalid value for daemons: {daemons}")

    replayed_processes = []
    for d in daemons:
      cfg = get_process_config(d)
      replayed_processes.append(cfg)
  else:
    replayed_processes = CONFIGS

  print("Replayed processes:", [p.proc_name for p in replayed_processes])
  print("\n\n", "*"*30, "\n\n", sep="")

  output_logs = replay_process(replayed_processes, all_msgs, frs, return_all_logs=True, custom_params=custom_params, disable_progress=disable_tqdm)

  return output_logs


def setup_data_readers(route: str, sidx: int, use_route_meta: bool) -> Tuple[LogReader, Dict[str, Any]]:
  if use_route_meta:
    r = Route(route)
    lr = LogReader(r.log_paths()[sidx])
    frs = {}
    if len(r.camera_paths()) > sidx and r.camera_paths()[sidx] is not None:
      frs['roadCameraState'] = FrameReader(r.camera_paths()[sidx])
    if len(r.ecamera_paths()) > sidx and r.ecamera_paths()[sidx] is not None:
      frs['wideCameraState'] = FrameReader(r.ecamera_paths()[sidx])
    if len(r.dcamera_paths()) > sidx and r.dcamera_paths()[sidx] is not None:
      frs['driverCameraState'] = FrameReader(r.dcamera_paths()[sidx])
  else:
    lr = LogReader(f"cd:/{route.replace('|', '/')}/{sidx}/rlog.bz2")
    frs = {
      'roadCameraState': FrameReader(f"cd:/{route.replace('|', '/')}/{sidx}/fcamera.hevc"),
      'driverCameraState': FrameReader(f"cd:/{route.replace('|', '/')}/{sidx}/dcamera.hevc"),
    }
    if next((True for m in lr if m.which() == "wideRoadCameraState"), False):
      frs['wideRoadCameraState'] = FrameReader(f"cd:/{route.replace('|', '/')}/{sidx}/ecamera.hevc")

  return lr, frs


def regen_and_save(
  route: str, sidx: int, daemons: Union[str, Iterable[str]] = "all", outdir: str = FAKEDATA,
  upload: bool = False, use_route_meta: bool = False, disable_tqdm: bool = False
) -> str:
  lr, frs = setup_data_readers(route, sidx, use_route_meta)
  output_logs = regen_segment(lr, frs, daemons, disable_tqdm=disable_tqdm)

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
  daemons = [p for p in args.whitelist_procs if p not in blacklist_set]
  regen_and_save(args.route, args.seg, daemons=daemons, upload=args.upload, outdir=args.outdir)
