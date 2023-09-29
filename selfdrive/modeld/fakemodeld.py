import time
from pathlib import Path

from cereal import log
from cereal.messaging import PubMaster
from cereal.visionipc import VisionIpcClient, VisionStreamType
from openpilot.system.swaglog import cloudlog


def main():
  vipc_client = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_ROAD, True)
  while not vipc_client.connect(False):
    time.sleep(0.1)

  cloudlog.warning("connected to main cam")

  pm = PubMaster(["modelV2", "cameraOdometry"])

  # Load some fixed modelV2/cameraOdometry messages
  with open(Path(__file__).parent / 'messages/modelV2.raw', 'rb') as f:
    modelV2_msg = f.read()
  with open(Path(__file__).parent / 'messages/cameraOdometry.raw', 'rb') as f:
    with log.Event.from_bytes(f.read()) as msg:
      cameraOdometry_msg = msg.as_builder()  # convert to builder so we can change the logMonoTime

  while True:
    # Wait for a frame before publishing
    buf = vipc_client.recv()
    if buf is None:
      continue

    # This isn't strictly necessary, it's just to clear some warnings from locationd
    cameraOdometry_msg.clear_write_flag()
    cameraOdometry_msg.logMonoTime = vipc_client.timestamp_eof

    pm.send("modelV2", modelV2_msg)
    pm.send("cameraOdometry", cameraOdometry_msg)


if __name__ == '__main__':
  main()
