#!/usr/bin/env python3
import os
import time
import cProfile
import pyray as rl
import numpy as np

from msgq.visionipc import VisionIpcServer, VisionStreamType
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.selfdrive.ui.mici.layouts.main import MiciMainLayout
from openpilot.system.ui.lib.application import gui_app
from openpilot.tools.lib.logreader import LogReader

FPS = 60


def chunk_messages_by_time(messages):
  dt_ns = 1e9 / FPS
  chunks = []
  current_services = {}
  next_time = messages[0].logMonoTime + dt_ns if messages else 0

  for msg in messages:
    if msg.logMonoTime >= next_time:
      chunks.append(current_services)
      current_services = {}
      next_time += dt_ns * ((msg.logMonoTime - next_time) // dt_ns + 1)
    current_services[msg.which()] = msg

  if current_services:
    chunks.append(current_services)
  return chunks


def patch_submaster(message_chunks):
  def mock_update(timeout=None):
    sm = ui_state.sm
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
  ui_state.sm.update = mock_update


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Profile openpilot UI rendering and state updates')
  parser.add_argument('route', type=str, nargs='?', default="302bab07c1511180/00000006--0b9a7005f1/3",
                      help='Route to use for profiling')
  parser.add_argument('--loop', type=int, default=1,
                      help='Number of times to loop the log (default: 1)')
  parser.add_argument('--output', type=str, default='cachegrind.out.ui',
                      help='Output file prefix (default: cachegrind.out.ui)')
  parser.add_argument('--max-seconds', type=float, default=None,
                      help='Maximum seconds of messages to process (default: all)')
  parser.add_argument('--headless', action='store_true',
                      help='Run in headless mode without GPU (for CI/testing)')
  args = parser.parse_args()

  print(f"Loading log from {args.route}...")
  lr = LogReader(args.route, sort_by_time=True)
  messages = list(lr) * args.loop

  print("Chunking messages...")
  message_chunks = chunk_messages_by_time(messages)
  if args.max_seconds:
    message_chunks = message_chunks[:int(args.max_seconds * FPS)]

  print("Initializing UI with GPU rendering...")

  if args.headless:
    os.environ['SDL_VIDEODRIVER'] = 'dummy'

  gui_app.init_window("UI Profiling", fps=600)
  main_layout = MiciMainLayout()
  main_layout.set_rect(rl.Rectangle(0, 0, gui_app.width, gui_app.height))

  print("Running...")
  patch_submaster(message_chunks)

  W, H = 2048, 1216
  vipc = VisionIpcServer("camerad")
  vipc.create_buffers(VisionStreamType.VISION_STREAM_ROAD, 5, W, H)
  vipc.start_listener()
  yuv_buffer_size = W * H + (W // 2) * (H // 2) * 2
  yuv_data = np.random.randint(0, 256, yuv_buffer_size, dtype=np.uint8).tobytes()
  with cProfile.Profile() as pr:
    for should_render in gui_app.render():
      if ui_state.sm.frame >= len(message_chunks):
        break
      if ui_state.sm.frame % 3 == 0:
        eof = int((ui_state.sm.frame % 3) * 0.05 * 1e9)
        vipc.send(VisionStreamType.VISION_STREAM_ROAD, yuv_data, ui_state.sm.frame % 3, eof, eof)
      ui_state.update()
      if should_render:
        main_layout.render()
    pr.dump_stats(f'{args.output}_deterministic.stats')

  rl.close_window()
  print("\nProfiling complete!")
  print(f"  run: python -m pstats {args.output}_deterministic.stats")
