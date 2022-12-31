from aiortc import VideoStreamTrack
import asyncio
import numpy as np
import av
import os
import sys
import time
import cereal.messaging as messaging

W, H = 1928, 1208
V4L2_BUF_FLAG_KEYFRAME = 8

class VisionIpcTrack(VideoStreamTrack):
  def __init__(self, sock_name, addr):
    super().__init__()
    self.codec = av.CodecContext.create("hevc", "r")
    os.environ["ZMQ"] = "1"
    messaging.context = messaging.Context()
    self.sock = messaging.sub_sock(sock_name, None, addr=addr, conflate=False)
    self.cnt = 0
    self.last_idx = -1
    self.seen_iframe = False
    self.time_q = []
    self.sock_name = sock_name

  async def recv(self):
    pts, time_base = await self.next_timestamp()
    frame = None
    while frame is None:
      msgs = messaging.drain_sock(self.sock, wait_for_one=True)
      for evt in msgs:
        evta = getattr(evt, evt.which())
        if evta.idx.encodeId != 0 and evta.idx.encodeId != (self.last_idx+1):
          print("DROP PACKET!")
        self.last_idx = evta.idx.encodeId
        if not self.seen_iframe and not (evta.idx.flags & V4L2_BUF_FLAG_KEYFRAME):
          print("waiting for iframe")
          continue
        self.time_q.append(time.monotonic())

        # put in header (first)
        if not self.seen_iframe:
          self.codec.decode(av.packet.Packet(evta.header))
          self.seen_iframe = True

        frames = self.codec.decode(av.packet.Packet(evta.data))
        if len(frames) == 0:
          print("DROP SURFACE")
          continue
        assert len(frames) == 1

        frame = frames[0]
        frame.pts = pts
        frame.time_base = time_base
    return frame

if __name__ == "__main__":
    from time import time_ns
    import sys

    async def test():
        frame_count=0
        start_time=time_ns()
        track = VisionIpcTrack("roadEncodeData", "192.168.88.249")
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