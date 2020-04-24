#!/usr/bin/env python3

import argparse
import time
import os

from tqdm import tqdm

from cereal.messaging import PubMaster, recv_one, sub_sock
from cereal.services import service_list
from tools.lib.logreader import LogReader
from xx.chffr.lib.route import Route, RouteSegment
from tools.lib.route_framereader import RouteFrameReader
from xx.uncommon.column_store import save_dict_as_column_store
from xx.pipeline.lib.log_time_series import append_dict
from selfdrive.test.process_replay.compare_logs import save_log

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run visiond on segment")
  parser.add_argument("segment_name", help="The segment to run")
  parser.add_argument("output_path", help="The output file")

  args = parser.parse_args()
  segment = RouteSegment.from_canonical_name(args.segment_name)
  route = Route(segment._name._route_name)

  frame_id_lookup = {}
  frame_reader = RouteFrameReader(route.camera_paths(), None, frame_id_lookup, readahead=True)

  msgs = list(LogReader(segment.log_path))

  pm = PubMaster(['liveCalibration', 'frame'])
  model_sock = sub_sock('model')

  # Read encodeIdx
  for msg in msgs:
    if msg.which() == 'encodeIdx':
      frame_id_lookup[msg.encodeIdx.frameId] = (msg.encodeIdx.segmentNum, msg.encodeIdx.segmentId)

  # Send some livecalibration messages to initalize visiond
  for msg in msgs:
    if msg.which() == 'liveCalibration':
      pm.send('liveCalibration', msg.as_builder())

  time.sleep(1.0)
  values = {}

  out_msgs = []
  for msg in tqdm(msgs):
    w = msg.which()

    if w == 'liveCalibration':
      pm.send(w, msg.as_builder())

    if w == 'frame':
      msg = msg.as_builder()

      frame_id = msg.frame.frameId
      img = frame_reader.get(frame_id, pix_fmt="rgb24")[:,:,::-1]

      msg.frame.image = img.flatten().tobytes()
      pm.send(w, msg)

      model = recv_one(model_sock)
      model = model.as_builder()
      model.logMonoTime = 0
      model = model.as_reader()
      out_msgs.append(model)

  save_log(args.output_path, out_msgs)

      # tm = model.logMonoTime / 1.0e9
      # model = model.model
  #     append_dict("model/data/path", tm, model.path.to_dict(), values)
  #     append_dict("model/data/left_lane", tm, model.leftLane.to_dict(), values)
  #     append_dict("model/data/right_lane", tm, model.rightLane.to_dict(), values)
  #     append_dict("model/data/lead", tm, model.lead.to_dict(), values)

  # save_dict_as_column_store(values, os.path.join(args.output_path, "LiveVisionD", args.segment_name))
