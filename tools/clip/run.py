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
from openpilot.tools.lib.logreader import LogReader
from openpilot.tools.lib.framereader import FrameReader
# GUI imports moved to clip() function after DISPLAY is set

DEFAULT_OUTPUT = 'output.mp4'
DEMO_START = 90
DEMO_END = 105
DEMO_ROUTE = 'a2a0ccea32023010/2023-07-27--13-01-19'
FRAMERATE = 20
RESOLUTION = '2160x1080'

OPENPILOT_FONT = str(Path(BASEDIR, 'selfdrive/assets/fonts/Inter-Regular.ttf').resolve())

logger = logging.getLogger('clip.py')

SERVICES = [
    "modelV2", "controlsState", "liveCalibration", "radarState", "deviceState",
    "pandaStates", "carParams", "driverMonitoringState", "carState", "driverStateV2",
    "roadCameraState", "wideRoadCameraState", "managerState", "selfdriveState",
    "longitudinalPlan", "rawAudioData"
]

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

  if quality == 'high':
    camera_paths = route.camera_paths()
  else:
    camera_paths = route.qcamera_paths()

  # Get frame resolution from the first valid segment
  first_segment_path = next((p for p in camera_paths if p is not None), None)
  if not first_segment_path:
    raise RuntimeError("No camera segments found to determine resolution")
  temp_fr = FrameReader(first_segment_path)
  width, height = temp_fr.w, temp_fr.h
  del temp_fr

  duration = end - start
  bit_rate_kbps = int(round(target_mb * 8 * 1024 * 1024 / duration / 1000))

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

    # Import GUI components AFTER DISPLAY is set (GLFW reads DISPLAY at import time)
    from openpilot.system.ui.lib.application import gui_app
    from openpilot.selfdrive.ui.layouts.main import MainLayout
    from openpilot.selfdrive.ui.ui_state import ui_state
    ui_state.sm = MockSubMaster()

    try:
      # Initialize Python UI in headless mode
      logger.debug('initializing UI...')
      # Set environment to force headless/offscreen rendering
      os.environ.setdefault('HEADLESS', '1')

      # Force render texture creation by setting scale != 1.0
      original_scale = os.environ.pop('SCALE', None)
      os.environ['SCALE'] = '2.0'

      # Initialize window and create render texture
      gui_app.init_window("Clip Renderer", fps=FRAMERATE)
      if gui_app._render_texture is not None:
        rl.unload_render_texture(gui_app._render_texture)
      gui_app._render_texture = rl.load_render_texture(width, height)
      rl.set_texture_filter(gui_app._render_texture.texture, rl.TextureFilter.TEXTURE_FILTER_BILINEAR)

      # Initialize MainLayout
      main_layout = MainLayout()
      main_layout.set_rect(rl.Rectangle(0, 0, width, height))

      # Restore original scale
      if original_scale:
        os.environ['SCALE'] = original_scale
      else:
        os.environ.pop('SCALE', None)

      # Start ffmpeg with stdin pipe
      logger.info(f'recording in progress ({duration}s)...')
      ffmpeg_proc = Popen(ffmpeg_cmd, stdin=PIPE, env=env)

      try:
        current_segment = -1
        fr = None
        segment_duration_frames = 60 * FRAMERATE

        # Render loop (simplified to write raw frames)
        for i in range(int(duration * FRAMERATE)):
          frame_idx = int(start * FRAMERATE + i)
          segment_num = frame_idx // segment_duration_frames

          if segment_num != current_segment:
            current_segment = segment_num
            segment_path = camera_paths[current_segment]
            if segment_path is None:
              logger.warning(f"Segment {current_segment} is missing camera footage, skipping.")
              # Create a black frame to keep video timing correct
              black_frame = np.zeros((height, width, 3), dtype=np.uint8)
              assert ffmpeg_proc.stdin is not None
              ffmpeg_proc.stdin.write(black_frame.tobytes())
              ffmpeg_proc.stdin.flush()
              continue

            logger.info(f"Loading segment {current_segment}: {segment_path}")
            fr = FrameReader(segment_path, pix_fmt='rgb24')

          frame_in_segment_idx = frame_idx % segment_duration_frames
          frame = fr.get(frame_in_segment_idx)

          # Write raw frame directly to ffmpeg, bypassing UI rendering
          assert ffmpeg_proc.stdin is not None
          ffmpeg_proc.stdin.write(frame.tobytes())
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
