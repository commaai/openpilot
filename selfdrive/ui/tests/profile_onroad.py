#!/usr/bin/env python3
import os
import time
import cProfile
import pyray as rl
import numpy as np
from typing import Any

from msgq.visionipc import VisionIpcServer, VisionStreamType
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.selfdrive.ui.layouts.main import MainLayout
from openpilot.system.ui.lib.application import gui_app
from openpilot.tools.lib.logreader import LogReader
from openpilot.tools.plotjuggler.juggle import DEMO_ROUTE
from openpilot.selfdrive.test.profiling.lib import ReplayDone
import cereal.messaging as messaging

W, H = 1928, 1208
CAMERA_BUFFER_COUNT = 5


class VideoServer:
  def __init__(self):
    self.vipc_server = VisionIpcServer("camerad")
    self.vipc_server.create_buffers(VisionStreamType.VISION_STREAM_ROAD, CAMERA_BUFFER_COUNT, W, H)
    self.vipc_server.create_buffers(VisionStreamType.VISION_STREAM_WIDE_ROAD, CAMERA_BUFFER_COUNT, W, H)
    self.vipc_server.start_listener()

    self.pm = messaging.PubMaster(['roadCameraState', 'wideRoadCameraState'])
    self.frame_id = 0

    yuv_buffer_size = W * H + (W // 2) * (H // 2) * 2
    self.yuv_data = np.random.randint(0, 256, yuv_buffer_size, dtype=np.uint8).tobytes()

  def send_frame(self):
    eof = int(self.frame_id * 0.05 * 1e9)

    self.vipc_server.send(VisionStreamType.VISION_STREAM_ROAD, self.yuv_data, self.frame_id, eof, eof)
    msg = messaging.new_message('roadCameraState', valid=True)
    msg.roadCameraState.frameId = self.frame_id
    msg.roadCameraState.transform = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    self.pm.send('roadCameraState', msg)

    self.vipc_server.send(VisionStreamType.VISION_STREAM_WIDE_ROAD, self.yuv_data, self.frame_id, eof, eof)
    msg = messaging.new_message('wideRoadCameraState', valid=True)
    msg.wideRoadCameraState.frameId = self.frame_id
    msg.wideRoadCameraState.transform = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    self.pm.send('wideRoadCameraState', msg)

    self.frame_id += 1


def chunk_messages_by_time(messages: list[Any], fps: int = 60) -> list[dict[str, Any]]:
  dt_ns = 1e9 / fps
  chunks = []

  start_time = messages[0].logMonoTime
  current_time_window = start_time + dt_ns
  current_services = {}

  for msg in messages:
    if msg.logMonoTime >= current_time_window:
      if current_services:
        chunks.append(current_services)
        current_services = {}
      while msg.logMonoTime >= current_time_window:
        current_time_window += dt_ns

    service = msg.which()
    current_services[service] = msg

  if current_services:
    chunks.append(current_services)

  return chunks


def patch_submaster_for_replay(message_chunks: list[dict[str, Any]]):
  sm = ui_state.sm
  def mock_update(timeout=None):
    if sm.frame >= len(message_chunks):
      raise ReplayDone("All message chunks processed")

    sm.updated = dict.fromkeys(sm.services, False)
    current_time = time.monotonic()
    for service, msg in message_chunks[sm.frame].items():
      if service in sm.data:
        sm.seen[service] = True
        sm.updated[service] = True

        msg_builder = msg.as_builder()
        sm.data[service] = getattr(msg_builder, service)
        sm.logMonoTime[service] = msg.logMonoTime
        sm.recv_time[service] = current_time
        sm.recv_frame[service] = sm.frame
        sm.valid[service] = True
    sm.frame += 1
  sm.update = mock_update


def profile_ui(message_chunks: list[dict[str, Any]], main_layout, video_server, output_prefix: str = 'cachegrind.out.ui'):
  print("Running deterministic profiling...")
  patch_submaster_for_replay(message_chunks)

  with cProfile.Profile() as pr:
    try:
      for should_render in gui_app.render():
        if ui_state.sm.frame % 3 == 0:
          video_server.send_frame()
        ui_state.update()
        if should_render:
          main_layout.render()

    except ReplayDone:
      pass

  pr.dump_stats(f'{output_prefix}_deterministic.stats')
  print(f"Deterministic profile saved to {output_prefix}_deterministic.stats")


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Profile openpilot UI rendering and state updates')
  parser.add_argument('--route', type=str, default=DEMO_ROUTE + "/1",
                      help='Route to use for profiling')
  parser.add_argument('--fps', type=int, default=60,
                      help='Target FPS for message chunking (default: 60)')
  parser.add_argument('--loop', type=int, default=1,
                      help='Number of times to loop the log (default: 1)')
  parser.add_argument('--output', type=str, default='cachegrind.out.ui',
                      help='Output file prefix (default: cachegrind.out.ui)')
  parser.add_argument('--max-messages', type=int, default=None,
                      help='Maximum number of messages to process (default: all)')
  parser.add_argument('--headless', action='store_true',
                      help='Run in headless mode without GPU (for CI/testing)')
  args = parser.parse_args()

  print(f"Loading log from {args.route}...")
  lr = LogReader(args.route, sort_by_time=True)

  messages = list(lr)
  if args.max_messages:
    messages = messages[:args.max_messages]
  messages = messages * args.loop

  print(f"Chunking messages for {args.fps} FPS...")
  message_chunks = chunk_messages_by_time(messages, fps=args.fps)
  video_server = VideoServer()

  print("Initializing UI with GPU rendering...")

  if args.headless:
    os.environ['SDL_VIDEODRIVER'] = 'dummy'

  gui_app.init_window("UI Profiling")
  main_layout = MainLayout()
  main_layout.set_rect(rl.Rectangle(0, 0, gui_app.width, gui_app.height))

  print("Starting profiling...")
  profile_ui(message_chunks, main_layout, video_server, output_prefix=args.output)

  rl.close_window()

  print("\nProfiling complete!")
  print("You can analyze the results with:")
  print(f"  Python pstats: python -m pstats {args.output}_deterministic.stats")
