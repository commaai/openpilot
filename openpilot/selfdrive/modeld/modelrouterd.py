#!/usr/bin/env python3
import time

import openpilot.cereal.messaging as messaging
from openpilot.cereal.messaging import PubMaster, SubMaster
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.modeld.fill_model_msg import fill_driving_model_data


def main() -> None:
  params = Params()
  sm = SubMaster(["smallModelV2", "bigModelV2", "carControl", "managerState"])
  pm = PubMaster(["modelV2", "drivingModelData", "cameraOdometry"])
  model = "bigModelV2" if params.get_bool("UsbGpuActive") else "smallModelV2"
  big_failed = params.get_bool("UsbGpuFailed")
  last_big_time = time.monotonic() if model == "bigModelV2" else None
  stopped_count = 0
  last_frame_id = -1

  while True:
    sm.update()
    big_available = sm.all_checks(["bigModelV2"])
    if sm.updated["bigModelV2"] and big_available:
      last_big_time = time.monotonic()

    if sm.updated["managerState"]:
      proc = next(p for p in sm["managerState"].processes if p.name == "bigmodeld")
      stopped_count = stopped_count + 1 if proc.shouldBeRunning and not proc.running else 0

    timed_out = last_big_time is not None and time.monotonic() - last_big_time > 10 * DT_MDL
    if (timed_out or stopped_count >= 2) and not big_failed:
      big_failed = True
      params.put_bool("UsbGpuFailed", True)
      cloudlog.event("big_model_unavailable", error=True)

    if sm.updated["carControl"] and sm.all_checks(["carControl"]) and not sm["carControl"].enabled:
      big_failed |= params.get_bool("UsbGpuFailed")
      next_model = "bigModelV2" if big_available and not big_failed else "smallModelV2"
      if model != next_model:
        model = next_model
        params.put_bool("UsbGpuActive", model == "bigModelV2", block=True)

    if not sm.updated[model]:
      continue

    output = sm[model]
    frame_id = output.modelV2.frameId
    if frame_id <= last_frame_id:
      continue

    model_send = messaging.new_message("modelV2", valid=sm.valid[model], logMonoTime=sm.logMonoTime[model])
    model_send.modelV2 = output.modelV2
    drivingdata_send = messaging.new_message("drivingModelData", logMonoTime=sm.logMonoTime[model])
    fill_driving_model_data(drivingdata_send, model_send)
    posenet_send = messaging.new_message("cameraOdometry", valid=output.cameraOdometryValid, logMonoTime=sm.logMonoTime[model])
    posenet_send.cameraOdometry = output.cameraOdometry
    pm.send("modelV2", model_send)
    pm.send("drivingModelData", drivingdata_send)
    pm.send("cameraOdometry", posenet_send)
    last_frame_id = frame_id


if __name__ == "__main__":
  main()
