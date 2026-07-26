#!/usr/bin/env python3
import os
import time
import cProfile
import gc
from contextlib import nullcontext
from unittest.mock import patch
from openpilot.system.ui.lib import raylib as rl
import numpy as np

from msgq.visionipc import VisionIpcServer, VisionStreamType
from openpilot.common.hardware import TICI
from openpilot.common.realtime import Priority, config_realtime_process
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.selfdrive.ui.mici.layouts.main import MiciMainLayout
from openpilot.system.ui.lib.application import gui_app
from openpilot.common.params import Params
from openpilot.common.version import terms_version, training_version
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
  ui_state.sm.update = mock_update  # ty: ignore[invalid-assignment]  # profiling hook


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
  parser.add_argument('--skip-onboarding', action='store_true',
                      help='Render the normal home/on-road layout regardless of persisted onboarding state')
  parser.add_argument('--screenshot', type=str,
                      help='Write the final CPU-backend framebuffer to this PNG')
  parser.add_argument('--no-profile', action='store_true',
                      help='Disable cProfile to measure undistorted frame latency')
  parser.add_argument('--camera-every', type=int, default=3,
                      help='Send a new synthetic camera frame every N UI frames; 0 disables camera frames')
  parser.add_argument('--camera-buffers', type=int, default=5,
                      help='Number of synthetic VisionIPC camera buffers (default: 5)')
  args = parser.parse_args()

  print(f"Loading log from {args.route}...")
  lr = LogReader(args.route, sort_by_time=True)
  messages = list(lr) * args.loop

  print("Chunking messages...")
  message_chunks = chunk_messages_by_time(messages)
  if args.max_seconds:
    message_chunks = message_chunks[:int(args.max_seconds * FPS)]

  print(f"Initializing UI with {'CPU' if rl.using_cpu_backend() else 'GPU'} rendering...")

  if args.headless:
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
  if TICI:
    # Match ui.py's steady-state placement. The CPU camera worker explicitly
    # drops this realtime policy and moves itself to core 4.
    config_realtime_process(5, Priority.CTRL_HIGH)

  gui_app.init_window("UI Profiling", fps=FPS)
  if args.skip_onboarding:
    original_get = Params.get

    def accepted_get(self, key, *get_args, **get_kwargs):
      if key == "HasAcceptedTerms":
        return terms_version
      if key == "CompletedTrainingVersion":
        return training_version
      return original_get(self, key, *get_args, **get_kwargs)

    with patch.object(Params, "get", accepted_get):
      main_layout = MiciMainLayout()
  else:
    main_layout = MiciMainLayout()

  print("Running...")
  patch_submaster(message_chunks)

  W, H = 2048, 1216
  vipc = VisionIpcServer("camerad")
  vipc.create_buffers(VisionStreamType.VISION_STREAM_ROAD, args.camera_buffers, W, H)
  vipc.start_listener()
  yuv_buffer_size = W * H + (W // 2) * (H // 2) * 2
  yuv_data = np.random.default_rng(0).integers(0, 256, yuv_buffer_size, dtype=np.uint8).tobytes()
  profiler = nullcontext() if args.no_profile else cProfile.Profile()
  render_times = []
  update_times = []
  frame_timestamps = []
  frame_kinds = []
  run_start = time.monotonic()
  process_cpu_start = time.process_time()
  with profiler as pr:
    for frame, (_, _, cpu_time) in enumerate(gui_app.render()):
      frame_timestamps.append(time.monotonic())
      render_times.append(cpu_time * 1000)
      frame_kinds.append((
        bool(ui_state.sm.updated.get("modelV2", False)),
        getattr(main_layout._car_onroad_layout.frame, "idx", -1),
      ))
      if args.skip_onboarding and frame == 1:
        main_layout._scroller.scroll_to(main_layout._car_onroad_layout.rect.x, smooth=False)
      if ui_state.sm.frame >= len(message_chunks):
        break
      if args.camera_every > 0 and ui_state.sm.frame % args.camera_every == 0:
        buffer_index = ui_state.sm.frame % args.camera_buffers
        eof = int(buffer_index * 0.05 * 1e9)
        vipc.send(VisionStreamType.VISION_STREAM_ROAD, yuv_data, buffer_index, eof, eof)
      update_start = time.monotonic()
      ui_state.update()
      update_times.append((time.monotonic() - update_start) * 1000)
    if not args.no_profile:
      pr.dump_stats(f'{args.output}_deterministic.stats')
  run_elapsed = time.monotonic() - run_start
  process_cpu_elapsed = time.process_time() - process_cpu_start

  if args.screenshot and rl.using_cpu_backend():
    from PIL import Image
    from openpilot.system.ui.lib.cpu_backend import framebuffer
    Image.fromarray(framebuffer(), "RGBA").save(args.screenshot)

  # Remove the camera plane before releasing the VisionIPC DMA-BUFs that may
  # still be referenced by the inline rotator.
  gui_app.close()
  main_layout._car_onroad_layout.close()
  vipc = None
  gc.collect()
  measured = np.asarray(render_times[min(30, len(render_times)):])
  if run_elapsed > 0:
    print(f"  cadence: frames={len(render_times)} elapsed={run_elapsed:.2f}s rate={len(render_times) / run_elapsed:.2f} FPS")
    print(f"  process CPU: {process_cpu_elapsed:.2f}s ({process_cpu_elapsed / run_elapsed * 100:.1f}% of one core)")
  cadence_warmup = min(30, max(0, len(frame_timestamps) - 2))
  if len(frame_timestamps) - cadence_warmup > 1:
    steady_elapsed = frame_timestamps[-1] - frame_timestamps[cadence_warmup]
    steady_frames = len(frame_timestamps) - cadence_warmup - 1
    cadence = f"frames={steady_frames} elapsed={steady_elapsed:.2f}s rate={steady_frames / steady_elapsed:.2f} FPS"
    print(f"  steady cadence: {cadence}")
  if update_times:
    updates = np.asarray(update_times)
    update_latency = f"mean={updates.mean():.2f} p50={np.percentile(updates, 50):.2f} "
    update_latency += f"p95={np.percentile(updates, 95):.2f} max={updates.max():.2f}"
    print(f"  state update ms: {update_latency}")
  if len(measured):
    latency = f"mean={measured.mean():.2f} p50={np.percentile(measured, 50):.2f} "
    latency += f"p95={np.percentile(measured, 95):.2f} max={measured.max():.2f}"
    print(f"  render ms: {latency}")
    warmup = min(30, len(render_times))
    kinds = frame_kinds[warmup:]
    camera_changed = [
      index == 0 or kinds[index][1] != kinds[index - 1][1]
      for index in range(len(kinds))
    ]
    for label, selector in (
      ("model update", lambda index: kinds[index][0]),
      ("no model", lambda index: not kinds[index][0]),
      ("camera update", lambda index: camera_changed[index]),
      ("cached camera", lambda index: not camera_changed[index]),
    ):
      subset = np.asarray([value for index, value in enumerate(measured) if selector(index)])
      if len(subset):
        print(f"  {label}: n={len(subset)} p50={np.percentile(subset, 50):.2f} p95={np.percentile(subset, 95):.2f}")
  if rl.using_cpu_backend() and os.getenv("CPU_RENDER_PROFILE") == "1":
    from openpilot.system.ui.lib.cpu_backend import cpu_profile_stats
    for name, (count, p50, p95) in cpu_profile_stats().items():
      print(f"  {name}: n={count} p50={p50:.2f} p95={p95:.2f}")
  print("\nProfiling complete!")
  if not args.no_profile:
    print(f"  run: python -m pstats {args.output}_deterministic.stats")
