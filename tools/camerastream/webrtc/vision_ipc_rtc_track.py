from cereal.visionipc.visionipc_pyx import VisionIpcClient # pylint: disable=no-name-in-module, import-error
from aiortc import VideoStreamTrack
import asyncio
from av import VideoFrame
import numpy as np

class VisionIpcTrack(VideoStreamTrack):
  def __init__(self, vision_stream_type):
    super().__init__()
    self.vipc_client = VisionIpcClient("camerad", vision_stream_type, True)

  async def recv(self):
    pts, time_base = await self.next_timestamp()

    # Connect if not connected
    while not self.vipc_client.is_connected():
      self.vipc_client.connect(True)
      print("vision ipc connected")

    raw_frame = None
    while raw_frame is None or not raw_frame.any():
      raw_frame = self.vipc_client.recv()

    raw_frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((self.vipc_client.height, self.vipc_client.width, 3))
    frame = VideoFrame.from_ndarray(raw_frame, "bgr24")
    frame.pts = pts
    frame.time_base = time_base

    return frame

if __name__ == "__main__":
    from time import time_ns
    import sys
    from cereal.visionipc.visionipc_pyx import VisionStreamType # pylint: disable=no-name-in-module, import-error

    async def test():
        frame_count=0
        start_time=time_ns()
        track = VisionIpcTrack(VisionStreamType.VISION_STREAM_WIDE_ROAD)
        while True:
            await track.recv()
            now = time_ns()
            playtime = now - start_time
            playtime_sec = playtime * 0.000000001
            if playtime_sec >= 1:
                print(f'fps: {frame_count}')
                frame_count = 0
                start_time = time_ns()
            else:
                frame_count+=1

    # Run event loop
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(test())
    except KeyboardInterrupt:
        sys.exit(0)