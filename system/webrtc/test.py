import time


start = time.monotonic()

from aiortc.mediastreams import VideoStreamTrack
from openpilot.system.webrtc.device.video import LiveStreamVideoStreamTrack
from teleoprtc import WebRTCAnswerBuilder

end = time.monotonic()

print((end - start) * 1000)