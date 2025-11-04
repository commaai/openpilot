#!/usr/bin/env python3

import logging
import os
import platform
import shutil
import sys
import time
import cffi
from argparse import ArgumentParser, ArgumentTypeError
from collections.abc import Sequence
from pathlib import Path
from random import randint
from subprocess import Popen, DEVNULL, PIPE
from typing import Literal

import pyray as rl
import numpy as np

from cereal import messaging
from openpilot.common.basedir import BASEDIR
from openpilot.common.params import Params, UnknownKeyName
from openpilot.common.prefix import OpenpilotPrefix
from openpilot.tools.lib.route import Route
from selfdrive.test.process_replay.migration import migrate_all
from openpilot.tools.lib.logreader import LogReader
from openpilot.tools.lib.framereader import FrameReader
from openpilot.common.transformations.orientation import rot_from_euler
# GUI imports moved to clip() function after DISPLAY is set

DEFAULT_OUTPUT = 'output.mp4'
DEMO_START = 90
DEMO_END = 105
DEMO_ROUTE = 'a2a0ccea32023010/2023-07-27--13-01-19'
FRAMERATE = 20
# RESOLUTION = os.environ.get("RESOLUTION", "2160x1080") # Dynamically set based on camera frame

OPENPILOT_FONT = str(Path(BASEDIR, 'selfdrive/assets/fonts/Inter-Regular.ttf').resolve())

logger = logging.getLogger('clip.py')

# SERVICES = [
#     "modelV2", "controlsState", "liveCalibration", "radarState", "deviceState",
#     "pandaStates", "carParams", "driverMonitoringState", "carState", "driverStateV2",
#     "roadCameraState", "wideRoadCameraState", "managerState", "selfdriveState",
#     "longitudinalPlan", "rawAudioData"
# ]

class MockSubMaster:
  def __init__(self):
    self.data = {}
    self.updated = {s: False for s in SERVICES}
    self.valid = {s: False for s in SERVICES}
    self.alive = {s: False for s in SERVICES}

  def __getitem__(self, s):
    return self.data.get(s)

  def __setitem__(self, s, msg):
    self.data[s] = msg
    self.updated[s] = True
    self.valid[s] = True
    self.alive[s] = True



# Initialize cffi for OpenGL calls
_ffi = cffi.FFI()
_ffi.cdef("""
  void glReadPixels(int x, int y, int width, int height, unsigned int format, unsigned int type, void *data);
  void glBindFramebuffer(unsigned int target, unsigned int framebuffer);
""")
# Load OpenGL library explicitly (libGL.so on Linux)
import ctypes.util
if platform.system() == 'Linux':
  opengl_lib = ctypes.util.find_library('GL') or 'libGL.so.1'
  _opengl = _ffi.dlopen(opengl_lib)
else:
  _opengl = _ffi.dlopen(None)


def extract_frame_from_texture(render_texture: rl.RenderTexture, width: int, height: int) -> bytes:
  """Extract RGB24 pixel data from a RenderTexture.

  Args:
    render_texture: The RenderTexture to read from
    width: Width of the texture
    height: Height of the texture

  Returns:
    RGB24 pixel data as bytes (width * height * 3 bytes)
  """
  # Bind the framebuffer to read from it
  _opengl.glBindFramebuffer(0x8D40, render_texture.id)  # GL_FRAMEBUFFER = 0x8D40

  # Allocate buffer for RGBA data (OpenGL returns RGBA)
  rgba_size = width * height * 4
  rgba_buffer = _ffi.new("unsigned char[]", rgba_size)

  # Read pixels from framebuffer (RGBA format)
  _opengl.glReadPixels(0, 0, width, height, 0x1908, 0x1401, rgba_buffer)  # GL_RGBA = 0x1908, GL_UNSIGNED_BYTE = 0x1401

  # Unbind framebuffer
  _opengl.glBindFramebuffer(0x8D40, 0)

  # Convert RGBA to RGB24 and flip vertically using numpy (much faster)
  rgba_array = np.frombuffer(_ffi.buffer(rgba_buffer), dtype=np.uint8).reshape(height, width, 4)
  # Extract RGB channels and flip vertically in one operation
  rgb_array = rgba_array[::-1, :, :3].reshape(height * width * 3)
  return rgb_array.tobytes()




def escape_ffmpeg_text(value: str):
  special_chars = {',': '\\,', ':': '\\:', '=': '\\=', '[': '\\[', ']': '\\]'}
  value = value.replace('\\', '\\\\\\\\\\\\\\\\')
  for char, escaped in special_chars.items():
    value = value.replace(char, escaped)
  return value


def get_logreader(route: Route):
  return LogReader(route.qlog_paths()[0] if len(route.qlog_paths()) else route.name.canonical_name)


def get_meta_text(lr: LogReader, route: Route):
  init_data = lr.first('initData')
  car_params = lr.first('carParams')
  origin_parts = init_data.gitRemote.split('/')
  origin = origin_parts[3] if len(origin_parts) > 3 else 'unknown'
  return ', '.join([
    f"openpilot v{init_data.version}",
    f"route: {route.name.canonical_name}",
    f"car: {car_params.carFingerprint}",
    f"origin: {origin}",
    f"branch: {init_data.gitBranch}",
    f"commit: {init_data.gitCommit[:7]}",
    f"modified: {str(init_data.dirty).lower()}",
  ])


def parse_args(parser: ArgumentParser):
  args = parser.parse_args()
  if args.demo:
    args.route = DEMO_ROUTE
    if args.start is None or args.end is None:
      args.start = DEMO_START
      args.end = DEMO_END
  elif args.route.count('/') == 1:
    if args.start is None or args.end is None:
      parser.error('must provide both start and end if timing is not in the route ID')
  elif args.route.count('/') == 3:
    if args.start is not None or args.end is not None:
      parser.error('don\'t provide timing when including it in the route ID')
    parts = args.route.split('/')
    args.route = '/'.join(parts[:2])
    args.start = int(parts[2])
    args.end = int(parts[3])
  if args.end <= args.start:
    parser.error(f'end ({args.end}) must be greater than start ({args.start})')

  try:
    args.route = Route(args.route, data_dir=args.data_dir)
  except Exception as e:
    parser.error(f'failed to get route: {e}')

  # FIXME: length isn't exactly max segment seconds, simplify to replay exiting at end of data
  length = round(args.route.max_seg_number * 60)
  if args.start >= length:
    parser.error(f'start ({args.start}s) cannot be after end of route ({length}s)')
  if args.end > length:
    parser.error(f'end ({args.end}s) cannot be after end of route ({length}s)')

  return args


def populate_car_params(lr: LogReader):
  init_data = lr.first('initData')
  assert init_data is not None

  params = Params()
  entries = init_data.params.entries
  for cp in entries:
    key, value = cp.key, cp.value
    try:
      params.put(key, params.cpp2python(key, value))
    except UnknownKeyName:
      # forks of openpilot may have other Params keys configured. ignore these
      logger.warning(f"unknown Params key '{key}', skipping")
  logger.debug('persisted CarParams')


def validate_env(parser: ArgumentParser):
  # Check ffmpeg
  if shutil.which('ffmpeg') is None:
    parser.exit(1, 'clip.py: error: missing ffmpeg command, is it installed?\n')
  # Check Xvfb (needed for GLFW to create OpenGL context on Linux)
  if platform.system() == 'Linux' and shutil.which('Xvfb') is None:
    parser.exit(1, 'clip.py: error: missing Xvfb command, is it installed?\n')


def validate_output_file(output_file: str):
  if not output_file.endswith('.mp4'):
    raise ArgumentTypeError('output must be an mp4')
  return output_file


def validate_route(route: str):
  if route.count('/') not in (1, 3):
    raise ArgumentTypeError(f'route must include or exclude timing, example: {DEMO_ROUTE}')
  return route


def validate_title(title: str):
  if len(title) > 80:
    raise ArgumentTypeError('title must be no longer than 80 chars')
  return title




def clip(
  data_dir: str | None,
  quality: Literal['low', 'high'],
  prefix: str,
  route: Route,
  out: str,
  start: int,
  end: int,
  speed: int,
  target_mb: int,
  title: str | None,
):
  logger.info(f'clipping route {route.name.canonical_name}, start={start} end={end} quality={quality} target_filesize={target_mb}MB')
  lr = get_logreader(route)
  init_data = lr.first('initData')
  device_type = str(init_data.deviceType) if init_data.deviceType else 'tici'

  if quality == 'high':
    camera_paths = route.camera_paths()
  else:
    camera_paths = route.qcamera_paths()

  # Use UI resolution (2160x1080) instead of camera resolution for proper path visualization
  # This matches the replay system approach where all UI rendering happens at UI resolution
  width, height = 2160, 1080
  # Still get camera frame resolution for scaling calculations
  first_segment_path = next((p for p in camera_paths if p is not None), None)
  if not first_segment_path:
    raise RuntimeError("No camera segments found to determine resolution")
  temp_fr = FrameReader(first_segment_path)
  camera_width, camera_height = temp_fr.w, temp_fr.h
  del temp_fr

  duration = end - start
  bit_rate_kbps = int(round(target_mb * 8 * 1024 * 1024 / duration / 1000))

  view_frame_from_device_frame = np.array([
    [ 0.,  0.,  1.],
    [ 1.,  0.,  0.],
    [ 0.,  1.,  0.]
  ]).T
  INF_POINT = np.array([1000.0, 0.0, 0.0])

  box_style = 'box=1:boxcolor=black@0.33:boxborderw=7'
  meta_text = get_meta_text(lr, route)
  overlays = [
    # metadata overlay
    f"drawtext=text='{escape_ffmpeg_text(meta_text)}':fontfile={OPENPILOT_FONT}:fontcolor=white:fontsize=15:{box_style}:x=(w-text_w)/2:y=5.5:enable='between(t,1,5)'",
    # route time overlay
    f"drawtext=text='%{{eif\\:floor(({start}+t)/60)\\:d\\:2}}\\:%{{eif\\:mod({start}+t\\,60)\\:d\\:2}}':fontfile={OPENPILOT_FONT}:fontcolor=white:fontsize=24:{box_style}:x=w-text_w-38:y=38"
  ]
  if title:
    overlays.append(f"drawtext=text='{escape_ffmpeg_text(title)}':fontfile={OPENPILOT_FONT}:fontcolor=white:fontsize=32:{box_style}:x=(w-text_w)/2:y=53")

  if speed > 1:
    overlays += [
      f"setpts=PTS/{speed}",
      "fps=60",
    ]

  # ffmpeg command using rawvideo input from stdin
  ffmpeg_cmd = [
    'ffmpeg', '-y',
    '-f', 'rawvideo',
    '-pix_fmt', 'rgb24',
    '-s', f'{width}x{height}',
    '-r', str(FRAMERATE),
    '-i', 'pipe:0',
    '-c:v', 'libx264',
    '-maxrate', f'{bit_rate_kbps}k',
    '-bufsize', f'{bit_rate_kbps*2}k',
    '-crf', '23',
    '-filter:v', ','.join(overlays),
    '-preset', 'ultrafast',
    '-pix_fmt', 'yuv420p',
    '-movflags', '+faststart',
    '-f', 'mp4',
    '-t', str(duration),
    out,
  ]

  with OpenpilotPrefix(prefix, shared_download_cache=True):
    populate_car_params(lr)

    # Setup Xvfb for GLFW (Linux only - GLFW needs X11 display for OpenGL context)
    # MUST set DISPLAY before importing GUI components (GLFW reads DISPLAY at import time)
    xvfb_proc = None
    original_display = os.environ.get('DISPLAY')
    if platform.system() == 'Linux':
      # Check if existing DISPLAY is valid, create Xvfb if needed
      display = os.environ.get('DISPLAY')
      if not display or Popen(['xdpyinfo', '-display', display], stdout=DEVNULL, stderr=DEVNULL).wait() != 0:
        display = f':{randint(99, 999)}'
        xvfb_proc = Popen(['Xvfb', display, '-screen', '0', f'{width}x{height}x24'], stdout=DEVNULL, stderr=DEVNULL)
        # Wait for Xvfb to be ready (max 5s)
        for _ in range(50):
          if xvfb_proc.poll() is not None:
            raise RuntimeError(f'Xvfb failed to start (exit code {xvfb_proc.returncode})')
          if Popen(['xdpyinfo', '-display', display], stdout=DEVNULL, stderr=DEVNULL).wait() == 0:
            break
          time.sleep(0.1)
        else:
          raise RuntimeError('Xvfb failed to become ready within 5s')
      os.environ['DISPLAY'] = display

    env = os.environ.copy()

    # Enable simulation mode BEFORE importing UI to disable SubMaster frequency tracking
    os.environ['SIMULATION'] = '1'

    # In simulation, SubMaster frequency checks are bypassed. However, the implementation has a bug where
    # it checks the frequency before checking if simulation is enabled, leading to a crash.
    # To work around this, we replace the entire `update_msgs` method with a corrected version.
    def patched_update_msgs(self, cur_time, msgs):
        self.frame += 1
        self.updated = dict.fromkeys(self.services, False)
        for msg in msgs:
            if msg is None:
                continue

            s = msg.which()
            if s not in self.services:
                continue

            self.seen[s] = True
            self.updated[s] = True

            self.freq_tracker[s].record_recv_time(cur_time)
            self.recv_time[s] = cur_time
            self.recv_frame[s] = self.frame
            self.data[s] = getattr(msg, s)
            self.logMonoTime[s] = msg.logMonoTime
            self.valid[s] = msg.valid

        for s in self.static_freq_services:
            # alive if delay is within 10x the expected frequency; checks relaxed in simulator
            self.alive[s] = (cur_time - self.recv_time[s]) < (10. / messaging.SERVICE_LIST[s].frequency) or (self.seen[s] and self.simulation)
            # Corrected logic: check simulation flag first to short-circuit and avoid ZeroDivisionError
            self.freq_ok[s] = self.simulation or self.freq_tracker[s].valid
    messaging.SubMaster.update_msgs = patched_update_msgs

    messaging.SubMaster.update_msgs = patched_update_msgs

    # Import GUI components AFTER patching and DISPLAY is set
    from openpilot.system.ui.lib.application import gui_app
    from openpilot.selfdrive.ui.onroad.augmented_road_view import AugmentedRoadView
    from openpilot.selfdrive.ui import UI_BORDER_SIZE
    from openpilot.selfdrive.ui.ui_state import ui_state
    from openpilot.common.transformations.camera import DEVICE_CAMERAS

    try:
      # Initialize Python UI in headless mode
      logger.debug('initializing UI...')
      # Set environment to force headless/offscreen rendering
      os.environ.setdefault('HEADLESS', '1')

      # Force render texture creation by setting scale != 1.0
      original_scale = os.environ.pop('SCALE', None)
      os.environ['SCALE'] = '2.0'

      # Initialize window and create render texture
      # gui_app.init_window("Clip Renderer", fps=FRAMERATE) # Moved below after FrameReader init
      if gui_app._render_texture is not None:
        rl.unload_render_texture(gui_app._render_texture)
      gui_app.init_window("Clip Renderer", fps=FRAMERATE)
      gui_app._render_texture = rl.load_render_texture(width, height)
      rl.set_texture_filter(gui_app._render_texture.texture, rl.TextureFilter.TEXTURE_FILTER_BILINEAR)

      # Initialize AugmentedRoadView which contains all the UI renderers
      road_view = AugmentedRoadView()
      road_view.set_rect(rl.Rectangle(0, 0, width, height))

      # fr and gui_app.init_window moved below after fcamera_path is defined
      # Restore original scale
      if original_scale:
        os.environ['SCALE'] = original_scale
      else:
        os.environ.pop('SCALE', None)



      try:
        # Build message lookup by timestamp for efficient access
        logger.info('indexing log messages...')
        all_msgs = []
        for log_path in route.log_paths():
          if log_path is not None:
            lr = LogReader(log_path)
            all_msgs.extend(lr)

        all_msgs = migrate_all(all_msgs)
        if not all_msgs:
          raise RuntimeError("No messages found in logs")

        # Determine services present in the log
        services = set(m.which() for m in all_msgs)
        # Always include services the UI expects, even if not in log
        services.add('selfdriveState')

        # Filter for services that are valid in the current openpilot version
        valid_services = [s for s in messaging.SERVICE_LIST if s in services]

        # Re-initialize SubMaster with the correct services
        ui_state.sm = messaging.SubMaster(valid_services)

        # get start mono time from the first log message
        start_mono_time = all_msgs[0].logMonoTime

        # filter and sort messages by time
        logger.info(f"Filtering {len(all_msgs)} messages for time range {start}s to {end}s")
        start_filter_mono_time = start_mono_time + start * 1e9
        end_filter_mono_time = start_mono_time + end * 1e9
        logger.info(f"Time filter range (monotonic): {start_filter_mono_time} to {end_filter_mono_time}")

        messages_by_time = sorted([(msg.logMonoTime, msg) for msg in all_msgs if start_filter_mono_time <= msg.logMonoTime <= end_filter_mono_time],\
                                  key=lambda x: x[0])
        logger.info(f"Found {len(messages_by_time)} messages in the time range.")

        # --- DEBUG: Inspect liveCalibration messages ---
        calib_msgs = [m for t, m in messages_by_time if m.which() == 'liveCalibration']
        logger.info(f"Found {len(calib_msgs)} liveCalibration messages in the time range.")
        if calib_msgs:
            logger.info("--- Checking `valid` and `height` data of first 5 liveCalibration messages ---")
            for i, msg in enumerate(calib_msgs[:5]):
                calib_height = msg.liveCalibration.height if msg.liveCalibration else 'N/A'
                logger.info(f"  liveCalibration message {i}: valid={msg.valid}, height={calib_height}")
            logger.info("--------------------------------------------------------------------------")
        # --- END DEBUG ---

        # --- DEBUG: Inspect modelV2 messages ---
        model_msgs = [m for t, m in messages_by_time if m.which() == 'modelV2']
        logger.info(f"Found {len(model_msgs)} modelV2 messages in the time range.")
        if model_msgs:
            logger.info("--- Checking `valid` and `position` data of first 5 modelV2 messages ---")
            for i, msg in enumerate(model_msgs[:5]):
                pos_len = len(msg.modelV2.position.x) if msg.modelV2 and msg.modelV2.position else 0
                logger.info(f"  modelV2 message {i}: valid={msg.valid}, position_len={pos_len}")
            logger.info("--------------------------------------------------------------------")
        # --- END DEBUG ---

        camera_messages = [m for t, m in messages_by_time if m.which() == 'roadCameraState']
        logger.info(f"Found {len(camera_messages)} roadCameraState messages to process.")

        if not camera_messages:
          raise RuntimeError(f'no roadCameraState messages found in time range {start}-{end}s')

        first_cam_sensor = camera_messages[0].roadCameraState.sensor
        camera = DEVICE_CAMERAS[(device_type, str(first_cam_sensor))]
        intrinsic_matrix = camera.fcam.intrinsics




        # Prime the UI with the last known state before the clip starts
        prime_mono_time = start_mono_time + start * 1e9
        prime_services = {'liveCalibration', 'modelV2', 'carParams', 'selfdriveState'}
        prime_msgs = []
        for s in prime_services:
            # Find the last message for each service before the clip starts
            last_msg = next((m for m in reversed(all_msgs) if m.logMonoTime < prime_mono_time and m.which() == s), None)
            if last_msg:
                prime_msgs.append(last_msg)

        if prime_msgs:
            # Feed these last-known-good messages to the SubMaster to set the initial state
            ui_state.sm.update_msgs(prime_mono_time / 1e9, prime_msgs)

        # Start ffmpeg with stdin pipe
        logger.info(f'recording in progress ({duration}s)...')
        ffmpeg_proc = Popen(ffmpeg_cmd, stdin=PIPE, env=env)

        current_segment = -1
        fr = None
        segment_duration_frames = 60 * FRAMERATE
        msg_idx = 0

        # Manually set ui_state to started mode before the loop begins
        ui_state.started = True
        ui_state.ignition = True

        for i, camera_msg in enumerate(camera_messages):
          frame_mono_time = camera_msg.logMonoTime

          # feed all messages up to this camera frame
          messages_to_feed = []
          while msg_idx < len(messages_by_time) and messages_by_time[msg_idx][0] <= frame_mono_time:
            messages_to_feed.append(messages_by_time[msg_idx][1])
            msg_idx += 1

          if messages_to_feed:
            ui_state.sm.update_msgs(frame_mono_time / 1e9, messages_to_feed)

          ui_state._update_state() # Process new messages
          ui_state._update_status() # update status to reflect new state

          # Get camera frame
          frame_id = camera_msg.roadCameraState.frameId
          # Calculate segment number based on elapsed time, not frame_id
          elapsed_time = (frame_mono_time - start_mono_time) / 1e9
          segment_num = int((elapsed_time - start) // 60) + (start // 60)

          if segment_num != current_segment:
            current_segment = segment_num
            segment_path = camera_paths[current_segment]
            if segment_path is None:
              logger.warning(f"Segment {current_segment} is missing camera footage, skipping.")
              continue
            logger.info(f"Loading segment {current_segment}: {segment_path}")
            fr = FrameReader(segment_path, pix_fmt='rgb24')

          frame_in_segment_idx = frame_id % (FRAMERATE * 60)
          frame_np = fr.get(frame_in_segment_idx)
          if frame_np is None:
            logger.warning(f"Failed to get frame {frame_id}, skipping")
            continue

          # Convert numpy frame to pyray Image/Texture using actual camera frame dimensions
          frame_image = rl.Image()
          frame_image.data = rl.ffi.cast("void *", frame_np.ctypes.data)
          frame_image.width = camera_width  # Use actual camera frame width
          frame_image.height = camera_height  # Use actual camera frame height
          frame_image.mipmaps = 1
          frame_image.format = rl.PixelFormat.PIXELFORMAT_UNCOMPRESSED_R8G8B8
          frame_texture = rl.load_texture_from_image(frame_image)

          # Render to texture
          rl.begin_texture_mode(gui_app._render_texture)
          rl.clear_background(rl.BLACK)

          # Ported from AugmentedRoadView and CameraView
          zoom = 1.1  # Fixed zoom factor from AugmentedRoadView

          # Get calibration from liveCalibration
          calib = ui_state.sm['liveCalibration']
          if len(calib.rpyCalib) == 3:
            device_from_calib = rot_from_euler(calib.rpyCalib)
            calibration = view_frame_from_device_frame @ device_from_calib

            calib_transform = intrinsic_matrix @ calibration
            kep = calib_transform @ INF_POINT
            w, h = 1620, 1080 # render at 1.5 aspect ratio
            cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

            margin = 5
            max_x_offset = cx * zoom - w / 2 - margin
            max_y_offset = cy * zoom - h / 2 - margin

            if abs(kep[2]) > 1e-6:
              x_offset_aug = np.clip((kep[0] / kep[2] - cx) * zoom, -max_x_offset, max_x_offset)
              y_offset_aug = np.clip((kep[1] / kep[2] - cy) * zoom, -max_y_offset, max_y_offset)
            else:
              x_offset_aug, y_offset_aug = 0, 0
          else:
            x_offset_aug, y_offset_aug = 0, 0
            w, h = width, height
            cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

          # The transform matrix from AugmentedRoadView
          transform = np.array([
            [zoom * 2 * cx / w, 0, -x_offset_aug / w * 2],
            [0, zoom * 2 * cy / h, -y_offset_aug / h * 2],
            [0, 0, 1.0]
          ])

          # From CameraView._render
          rect = rl.Rectangle(0, 0, width, height)
          scale_x = rect.width * transform[0, 0]
          scale_y = rect.height * transform[1, 1]

          x_offset_cam = rect.x + (rect.width - scale_x) / 2
          y_offset_cam = rect.y + (rect.height - scale_y) / 2

          x_offset_cam += transform[0, 2] * rect.width / 2
          y_offset_cam += transform[1, 2] * rect.height / 2

          dst_rect = rl.Rectangle(x_offset_cam + (width - w) / 2, y_offset_cam, scale_x, scale_y)
          src_rect = rl.Rectangle(0, 0, camera_width, camera_height)

          # Draw camera frame zoomed to fill the screen
          rl.draw_texture_pro(frame_texture, src_rect, dst_rect, rl.Vector2(0, 0), 0.0, rl.WHITE)
          dst_rect = rl.Rectangle(0, 0, width, height)

          # Draw camera frame zoomed to fill the screen
          rl.draw_texture_pro(frame_texture, src_rect, dst_rect, rl.Vector2(0, 0), 0.0, rl.WHITE)

          # Define full rect and content rect (with border padding)
          from openpilot.selfdrive.ui import UI_BORDER_SIZE
          full_rect = rl.Rectangle(0, 0, width, height)
          content_rect = rl.Rectangle(
            UI_BORDER_SIZE,
            UI_BORDER_SIZE,
            width - 2 * UI_BORDER_SIZE,
            height - 2 * UI_BORDER_SIZE,
          )

          # Enable scissor mode to clip rendering within content rectangle
          rl.begin_scissor_mode(
            int(content_rect.x),
            int(content_rect.y),
            int(content_rect.width),
            int(content_rect.height)
          )

          road_view._update_calibration()
          road_view._content_rect = content_rect # Assign to instance variable

          # Manually calculate the frame matrix to set the transform for the model renderer
          road_view._calc_frame_matrix(content_rect) # Pass content_rect here

          # Render UI overlays on top using the properly calculated content rectangle
          road_view.model_renderer.render(road_view._content_rect)
          road_view._hud_renderer.render(road_view._content_rect)
          road_view.alert_renderer.render(road_view._content_rect)
          road_view.driver_state_renderer.render(road_view._content_rect)

          # End clipping region
          rl.end_scissor_mode()

          # --- DEBUG: Inspect ModelRenderer internal state ---
          if i % 20 == 0: # Log once per second
              renderer = road_view.model_renderer
              raw_size = renderer._path.raw_points.size
              proj_size = renderer._path.projected_points.size
              transform_flat = renderer._car_space_transform.flatten()
              logger.info(f"[Frame {i}] Renderer state: raw_points={raw_size}, projected_points={proj_size}")
              # Log first few elements of transform matrix to see if it's changing/valid
              logger.info(f"  Transform matrix (first 4): {transform_flat[:4]}")
          # --- END DEBUG ---

          # Draw colored border based on driving state
          road_view._draw_border(full_rect)

          rl.end_texture_mode()

          rl.unload_texture(frame_texture)

          # Extract frame pixels
          frame_data = extract_frame_from_texture(gui_app._render_texture, width, height)

          # Write to ffmpeg
          assert ffmpeg_proc.stdin is not None
          ffmpeg_proc.stdin.write(frame_data)
          ffmpeg_proc.stdin.flush()

        # Cleanup
        assert ffmpeg_proc.stdin is not None
        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()

        if ffmpeg_proc.returncode != 0:
          raise RuntimeError(f'ffmpeg failed with exit code {ffmpeg_proc.returncode}')

      finally:
        # Cleanup UI
        gui_app.close()

      logger.info(f'recording complete: {Path(out).resolve()}')
    finally:
      # Cleanup Xvfb and restore DISPLAY
      if xvfb_proc is not None:
        xvfb_proc.terminate()
        xvfb_proc.wait()
      # Restore original DISPLAY
      if original_display:
        os.environ['DISPLAY'] = original_display
      else:
        os.environ.pop('DISPLAY', None)


def main():
  p = ArgumentParser(prog='clip.py', description='clip your openpilot route.', epilog='comma.ai')
  validate_env(p)
  route_group = p.add_mutually_exclusive_group(required=True)
  route_group.add_argument('route', nargs='?', type=validate_route, help=f'The route (e.g. {DEMO_ROUTE} or {DEMO_ROUTE}/{DEMO_START}/{DEMO_END})')
  route_group.add_argument('--demo', help='use the demo route', action='store_true')
  p.add_argument('-d', '--data-dir', help='local directory where route data is stored')
  p.add_argument('-e', '--end', help='stop clipping at <end> seconds', type=int)
  p.add_argument('-f', '--file-size', help='target file size (Discord/GitHub support max 10MB, default is 9MB)', type=float, default=9.)
  p.add_argument('-o', '--output', help='output clip to (.mp4)', type=validate_output_file, default=DEFAULT_OUTPUT)
  p.add_argument('-p', '--prefix', help='openpilot prefix', default=f'clip_{randint(100, 99999)}')
  p.add_argument('-q', '--quality', help='quality of camera (low = qcam, high = hevc)', choices=['low', 'high'], default='high')
  p.add_argument('-x', '--speed', help='record the clip at this speed multiple', type=int, default=1)
  p.add_argument('-s', '--start', help='start clipping at <start> seconds', type=int)
  p.add_argument('-t', '--title', help='overlay this title on the video (e.g. "Chill driving across the Golden Gate Bridge")', type=validate_title)
  args = parse_args(p)
  exit_code = 1
  try:
    clip(
      data_dir=args.data_dir,
      quality=args.quality,
      prefix=args.prefix,
      route=args.route,
      out=args.output,
      start=args.start,
      end=args.end,
      speed=args.speed,
      target_mb=args.file_size,
      title=args.title,
    )
    exit_code = 0
  except KeyboardInterrupt as e:
    logger.exception('interrupted by user', exc_info=e)
  except Exception as e:
    logger.exception('encountered error', exc_info=e)
  sys.exit(exit_code)


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s\t%(message)s')
  main()
