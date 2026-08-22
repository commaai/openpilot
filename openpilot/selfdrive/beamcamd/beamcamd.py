import time
import numpy as np
from openpilot.cereal.visionipc import VisionStreamType
from openpilot.cereal import messaging
from openpilot.common.realtime import Ratekeeper
from msgq.visionipc import VisionIpcServer
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info

# dummy black screen for now

TR = [1.0, 0.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 0.0, 1.0]

def main():
  W, H = 1928, 1208

  server = VisionIpcServer("camerad")
  stride, y_height, uv_height, size = get_nv12_info(W, H)
  uv_offset = stride * y_height

  server.create_buffers_with_sizes(VisionStreamType.VISION_STREAM_WIDE_ROAD, 20, W, H, size, stride, uv_offset)
  server.create_buffers_with_sizes(VisionStreamType.VISION_STREAM_NARROW_ROAD, 20, W, H, size, stride, uv_offset)
  server.create_buffers_with_sizes(VisionStreamType.VISION_STREAM_CABIN, 4, W, H, size, stride, uv_offset)

  server.start_listener()

  pm = messaging.PubMaster(["wideRoadCameraState", "narrowRoadCameraState"])

  blank_yuv = np.zeros(size, dtype=np.uint8)
  # simple padding with 128 for UV doesn't matter much since its just a blank screen anyway
  blank_yuv[uv_offset:uv_offset + (stride * uv_height)] = 128
  blank_yuv_bytes = blank_yuv.tobytes()

  rate = Ratekeeper(20, print_delay_threshold=None)
  frame_id = 0

  # file sending behavior based off of system/camerad
  while True:
    timestamp = int(frame_id * 0.05 * 1e9)

    server.send(VisionStreamType.VISION_STREAM_WIDE_ROAD, blank_yuv_bytes, frame_id, timestamp, timestamp)
    server.send(VisionStreamType.VISION_STREAM_NARROW_ROAD, blank_yuv_bytes, frame_id, timestamp, timestamp)

    # update cereal (wide road)
    dat = messaging.new_message("wideRoadCameraState", valid=True)
    msg = {"frameId": frame_id, "transform": TR, "sensor": "unknown"}
    setattr(dat, "wideRoadCameraState", msg)
    pm.send("wideRoadCameraState", dat)

    # update cereal (narrow road)
    dat = messaging.new_message("narrowRoadCameraState", valid=True)
    msg = {"frameId": frame_id, "transform": TR, "sensor": "unknown"}
    setattr(dat, "narrowRoadCameraState", msg)
    pm.send("narrowRoadCameraState", dat)

    # update frame count / id
    frame_id += 1

    # wait for 20hz
    rate.keep_time()

if __name__ == "__main__":
  main()