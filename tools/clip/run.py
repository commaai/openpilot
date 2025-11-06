#!/usr/bin/env python3

import logging
import os
import platform
import shutil
import sys
import time
import cffi
import ctypes.util
from argparse import ArgumentParser, ArgumentTypeError
from collections.abc import Sequence
from contextlib import contextmanager
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

DEFAULT_OUTPUT = 'output.mp4'
DEMO_START = 90
DEMO_END = 105
DEMO_ROUTE = 'a2a0ccea32023010/2023-07-27--13-01-19'
FRAMERATE = 20
UI_WIDTH, UI_HEIGHT = 2160, 1080

OPENPILOT_FONT = str(Path(BASEDIR, 'selfdrive/assets/fonts/Inter-Regular.ttf').resolve())

logger = logging.getLogger('clip.py')

# Initialize cffi for OpenGL calls
_ffi = cffi.FFI()
_ffi.cdef("""
  void glReadPixels(int x, int y, int width, int height, unsigned int format, unsigned int type, void *data);
  void glBindFramebuffer(unsigned int target, unsigned int framebuffer);
""")
if platform.system() == 'Linux':
  opengl_lib = ctypes.util.find_library('GL') or 'libGL.so.1'
  _opengl = _ffi.dlopen(opengl_lib)
else:
  _opengl = _ffi.dlopen(None)


def extract_frame_from_texture(render_texture: rl.RenderTexture, width: int, height: int) -> bytes:
  _opengl.glBindFramebuffer(0x8D40, render_texture.id)
  rgba_size = width * height * 4
  rgba_buffer = _ffi.new("unsigned char[]", rgba_size)
  _opengl.glReadPixels(0, 0, width, height, 0x1908, 0x1401, rgba_buffer)
  _opengl.glBindFramebuffer(0x8D40, 0)
  rgba_array = np.frombuffer(_ffi.buffer(rgba_buffer), dtype=np.uint8).reshape(height, width, 4)
  rgb_array = rgba_array[::-1, :, :3].reshape(height * width * 3)
  return rgb_array.tobytes()


def escape_ffmpeg_text(value: str):
  special_chars = {',': r'\,', ':': r'\:', '=': r'\=', '[': r'\[', ']': r'\]'}
  value = value.replace('\\', '\\\\\\\\')
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

def populate_car_params(lr: LogReader):
  init_data = lr.first('initData')
  assert init_data is not None

  params = Params()
  for cp in init_data.params.entries:
    try:
      params.put(cp.key, params.cpp2python(cp.key, cp.value))
    except UnknownKeyName:
      logger.warning(f"unknown Params key '{cp.key}', skipping")

  car_params = lr.first('carParams')
  if car_params:
    params.put("CarParams", car_params.as_builder().to_bytes())
  logger.debug('persisted CarParams')

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
    self.alive[s] = (cur_time - self.recv_time[s]) < (10. / messaging.SERVICE_LIST[s].frequency) or (self.seen[s] and self.simulation)
    self.freq_ok[s] = self.simulation or self.freq_tracker[s].valid


class ClipGenerator:
  def __init__(self, **kwargs):
    self.args = kwargs
    self.lr = get_logreader(self.args['route'])
    self.init_data = self.lr.first('initData')
    self.device_type = str(self.init_data.deviceType) if self.init_data.deviceType else 'tici'

    if self.args['quality'] == 'high':
      self.camera_paths = self.args['route'].camera_paths()
    else:
      self.camera_paths = self.args['route'].qcamera_paths()

    first_segment_path = next((p for p in self.camera_paths if p is not None), None)
    if not first_segment_path:
      raise RuntimeError("No camera segments found to determine resolution")
    temp_fr = FrameReader(first_segment_path)
    self.camera_width, self.camera_height = temp_fr.w, temp_fr.h
    del temp_fr

  @contextmanager
  def _setup_environment(self):
    original_display = os.environ.get('DISPLAY')
    xvfb_proc = None
    if platform.system() == 'Linux':
      display = os.environ.get('DISPLAY')
      if not display or Popen(['xdpyinfo', '-display', display], stdout=DEVNULL, stderr=DEVNULL).wait() != 0:
        display = f':{randint(99, 999)}'
        xvfb_proc = Popen(['Xvfb', display, '-screen', '0', f'{UI_WIDTH}x{UI_HEIGHT}x24'], stdout=DEVNULL, stderr=DEVNULL)
        for _ in range(50):
          if xvfb_proc.poll() is not None:
            raise RuntimeError(f'Xvfb failed to start (exit code {xvfb_proc.returncode})')
          if Popen(['xdpyinfo', '-display', display], stdout=DEVNULL, stderr=DEVNULL).wait() == 0:
            break
          time.sleep(0.1)
        else:
          raise RuntimeError('Xvfb failed to become ready within 5s')
      os.environ['DISPLAY'] = display

    original_simulation = os.environ.get('SIMULATION')
    os.environ['SIMULATION'] = '1'
    original_update_msgs = messaging.SubMaster.update_msgs
    messaging.SubMaster.update_msgs = patched_update_msgs

    try:
      yield
    finally:
      if xvfb_proc:
        xvfb_proc.terminate()
        xvfb_proc.wait()
      if original_display:
        os.environ['DISPLAY'] = original_display
      else:
        os.environ.pop('DISPLAY', None)
      if original_simulation:
        os.environ['SIMULATION'] = original_simulation
      else:
        os.environ.pop('SIMULATION', None)
      messaging.SubMaster.update_msgs = original_update_msgs

  def _get_ffmpeg_cmd(self):
    duration = self.args['end'] - self.args['start']
    bit_rate_kbps = int(round(self.args['target_mb'] * 8 * 1024 * 1024 / duration / 1000))
    meta_text = get_meta_text(self.lr, self.args['route'])
    box_style = 'box=1:boxcolor=black@0.33:boxborderw=7'
    overlays = [
      f"drawtext=text='{escape_ffmpeg_text(meta_text)}':fontfile={OPENPILOT_FONT}:fontcolor=white:fontsize=15:{box_style}:x=(w-text_w)/2:y=5.5:enable='between(t,1,5)'",
            rf"drawtext=text='%{{eif\:floor(({self.args['start']}+t)/60)\:d\:2}}\:%{{eif\:mod({self.args['start']}+t,60)\:d\:2}}':fontfile={OPENPILOT_FONT}:fontcolor=white:fontsize=24:{box_style}:x=w-text_w-38:y=38"     ]
    if self.args['title']:
      overlays.append(f"drawtext=text='{escape_ffmpeg_text(self.args['title'])}':fontfile={OPENPILOT_FONT}:fontcolor=white:fontsize=32:{box_style}:x=(w-text_w)/2:y=53")
    if self.args['speed'] > 1:
      overlays += [f"setpts=PTS/{self.args['speed']}", "fps=60"]

    return [
      'ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{UI_WIDTH}x{UI_HEIGHT}',
      '-r', str(FRAMERATE), '-i', 'pipe:0', '-c:v', 'libx264', '-maxrate', f'{bit_rate_kbps}k',
      '-bufsize', f'{bit_rate_kbps*2}k', '-crf', '23', '-filter:v', ','.join(overlays),
      '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
      '-f', 'mp4', '-t', str(duration), self.args['out'],
    ]

  def run(self):
    logger.info(f"clipping route {self.args['route'].name.canonical_name}, start={self.args['start']} end={self.args['end']} quality={self.args['quality']} target_filesize={self.args['target_mb']}MB")

    with OpenpilotPrefix(self.args['prefix'], shared_download_cache=True):
      populate_car_params(self.lr)
      with self._setup_environment():
        from openpilot.system.ui.lib.application import gui_app
        from openpilot.selfdrive.ui.onroad.augmented_road_view import AugmentedRoadView
        from openpilot.selfdrive.ui import UI_BORDER_SIZE
        from openpilot.selfdrive.ui.ui_state import ui_state
        from openpilot.common.transformations.camera import DEVICE_CAMERAS

        try:
          os.environ.setdefault('HEADLESS', '1')
          original_scale = os.environ.pop('SCALE', None)
          os.environ['SCALE'] = '2.0'
          gui_app.init_window("Clip Renderer", fps=FRAMERATE)
          gui_app._render_texture = rl.load_render_texture(UI_WIDTH, UI_HEIGHT)
          rl.set_texture_filter(gui_app._render_texture.texture, rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
          if original_scale:
            os.environ['SCALE'] = original_scale
          else:
            os.environ.pop('SCALE', None)

          road_view = AugmentedRoadView()
          road_view.set_rect(rl.Rectangle(0, 0, UI_WIDTH, UI_HEIGHT))

          all_msgs = migrate_all([m for p in self.args['route'].log_paths() if p for m in LogReader(p)])
          if not all_msgs:
            raise RuntimeError("No messages found in logs")

          services = {m.which() for m in all_msgs} | {'selfdriveState', 'radarState', 'longitudinalPlan'}
          valid_services = [s for s in messaging.SERVICE_LIST if s in services]
          ui_state.sm = messaging.SubMaster(valid_services)

          start_mono_time = all_msgs[0].logMonoTime
          start_filter_mono_time = start_mono_time + self.args['start'] * 1e9
          end_filter_mono_time = start_mono_time + self.args['end'] * 1e9
          messages_by_time = sorted([(m.logMonoTime, m) for m in all_msgs if start_filter_mono_time <= m.logMonoTime <= end_filter_mono_time], key=lambda x: x[0])
          camera_messages = [m for t, m in messages_by_time if m.which() == 'roadCameraState']
          if not camera_messages:
            raise RuntimeError(f"no roadCameraState messages found in time range {self.args['start']}-{self.args['end']}s")

          first_cam_sensor = camera_messages[0].roadCameraState.sensor
          camera = DEVICE_CAMERAS[(self.device_type, str(first_cam_sensor))]
          intrinsic_matrix = camera.fcam.intrinsics

          prime_mono_time = start_mono_time + self.args['start'] * 1e9
          prime_services = {'liveCalibration', 'modelV2', 'carParams', 'selfdriveState', 'radarState'}
          prime_msgs = [m for s in prime_services for m in [next((m for m in reversed(all_msgs) if m.logMonoTime < prime_mono_time and m.which() == s), None)] if m]
          if prime_msgs:
            ui_state.sm.update_msgs(prime_mono_time / 1e9, prime_msgs)

          ffmpeg_proc = Popen(self._get_ffmpeg_cmd(), stdin=PIPE, env=os.environ.copy())
          current_segment, fr, msg_idx = -1, None, 0
          ui_state.started = ui_state.ignition = True

          for i, camera_msg in enumerate(camera_messages):
            frame_mono_time = camera_msg.logMonoTime
            messages_to_feed = []
            while msg_idx < len(messages_by_time) and messages_by_time[msg_idx][0] <= frame_mono_time:
              messages_to_feed.append(messages_by_time[msg_idx][1])
              msg_idx += 1
            if messages_to_feed:
              ui_state.sm.update_msgs(frame_mono_time / 1e9, messages_to_feed)
            ui_state._update_state()
            ui_state._update_status()

            elapsed_time = (frame_mono_time - start_mono_time) / 1e9
            segment_num = int((elapsed_time - self.args['start']) // 60) + (self.args['start'] // 60)
            if segment_num != current_segment:
              current_segment = segment_num
              if self.camera_paths[current_segment] is not None:
                fr = FrameReader(self.camera_paths[current_segment], pix_fmt='rgb24')
            if fr is None:
              continue

            frame_in_segment_idx = camera_msg.roadCameraState.frameId % (FRAMERATE * 60)
            frame_np = fr.get(frame_in_segment_idx)
            if frame_np is None:
              continue

            frame_image = rl.Image()
            frame_image.data = rl.ffi.cast("void *", frame_np.ctypes.data)
            frame_image.width = self.camera_width
            frame_image.height = self.camera_height
            frame_image.mipmaps = 1
            frame_image.format = rl.PixelFormat.PIXELFORMAT_UNCOMPRESSED_R8G8B8
            frame_texture = rl.load_texture_from_image(frame_image)

            rl.begin_texture_mode(gui_app._render_texture)
            rl.clear_background(rl.BLACK)

            calib = ui_state.sm['liveCalibration']
            if len(calib.rpyCalib) == 3:
                device_from_calib = rot_from_euler(calib.rpyCalib)
                view_frame_from_device_frame = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]]).T
                calibration = view_frame_from_device_frame @ device_from_calib
                calib_transform = intrinsic_matrix @ calibration
                kep = calib_transform @ np.array([1000.0, 0.0, 0.0])
                w, h, zoom = UI_WIDTH, UI_HEIGHT, 1.1
                cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
                max_x_offset, max_y_offset = cx * zoom - w / 2 - 5, cy * zoom - h / 2 - 5
                x_offset_aug = np.clip((kep[0] / kep[2] - cx) * zoom, -max_x_offset, max_x_offset) if abs(kep[2]) > 1e-6 else 0
                y_offset_aug = np.clip((kep[1] / kep[2] - cy) * zoom, -max_y_offset, max_y_offset) if abs(kep[2]) > 1e-6 else 0
            else:
                x_offset_aug, y_offset_aug = 0, 0

            transform = np.array([[zoom * 2 * cx / w, 0, -x_offset_aug / w * 2], [0, zoom * 2 * cy / h, -y_offset_aug / h * 2], [0, 0, 1.0]])
            scale_x, scale_y = UI_WIDTH * transform[0, 0], UI_HEIGHT * transform[1, 1]
            x_offset, y_offset = (UI_WIDTH - scale_x) / 2 + transform[0, 2] * UI_WIDTH / 2, (UI_HEIGHT - scale_y) / 2 + transform[1, 2] * UI_HEIGHT / 2
            if scale_x / UI_WIDTH > scale_y / UI_HEIGHT:
                final_scale_y = scale_y * (UI_WIDTH * transform[0, 0]) / scale_x
                final_x_offset, final_y_offset = x_offset, (UI_HEIGHT - final_scale_y) / 2
            else:
                final_scale_x = scale_x * (UI_HEIGHT * transform[1, 1]) / scale_y
                final_x_offset, final_y_offset = (UI_WIDTH - final_scale_x) / 2, y_offset
            dst_rect = rl.Rectangle(final_x_offset, final_y_offset, scale_x, scale_y)
            src_rect = rl.Rectangle(0, 0, self.camera_width, self.camera_height)
            rl.draw_texture_pro(frame_texture, src_rect, dst_rect, rl.Vector2(0, 0), 0.0, rl.WHITE)

            content_rect = rl.Rectangle(UI_BORDER_SIZE, UI_BORDER_SIZE, UI_WIDTH - 2 * UI_BORDER_SIZE, UI_HEIGHT - 2 * UI_BORDER_SIZE)
            rl.begin_scissor_mode(int(content_rect.x), int(content_rect.y), int(content_rect.width), int(content_rect.height))
            road_view._update_calibration()
            road_view._content_rect = content_rect
            road_view._calc_frame_matrix(content_rect)
            road_view.model_renderer.render(road_view._content_rect)
            road_view._hud_renderer.render(road_view._content_rect)
            road_view.alert_renderer.render(road_view._content_rect)
            road_view.driver_state_renderer.render(road_view._content_rect)
            rl.end_scissor_mode()
            road_view._draw_border(rl.Rectangle(0, 0, UI_WIDTH, UI_HEIGHT))
            rl.end_texture_mode()
            rl.unload_texture(frame_texture)

            frame_data = extract_frame_from_texture(gui_app._render_texture, UI_WIDTH, UI_HEIGHT)
            ffmpeg_proc.stdin.write(frame_data)
            ffmpeg_proc.stdin.flush()

          ffmpeg_proc.stdin.close()
          ffmpeg_proc.wait()
          if ffmpeg_proc.returncode != 0:
            raise RuntimeError(f'ffmpeg failed with exit code {ffmpeg_proc.returncode}')
        finally:
          gui_app.close()
      logger.info(f'recording complete: {Path(self.args["out"]).resolve()}')

def clip(**kwargs):
  c = ClipGenerator(**kwargs)
  c.run()

def parse_args(parser: ArgumentParser):
  args = parser.parse_args()
  if args.demo:
    args.route = DEMO_ROUTE
    if args.start is None or args.end is None:
      args.start, args.end = DEMO_START, DEMO_END
  elif args.route.count('/') == 1:
    if args.start is None or args.end is None:
      parser.error('must provide both start and end if timing is not in the route ID')
  elif args.route.count('/') == 3:
    if args.start is not None or args.end is not None:
      parser.error('don\'t provide timing when including it in the route ID')
    parts = args.route.split('/')
    args.route, args.start, args.end = '/'.join(parts[:2]), int(parts[2]), int(parts[3])
  if args.end <= args.start:
    parser.error(f'end ({args.end}) must be greater than start ({args.start})')

  try:
    args.route = Route(args.route, data_dir=args.data_dir)
  except Exception as e:
    parser.error(f'failed to get route: {e}')

  length = round(args.route.max_seg_number * 60)
  if args.start >= length or args.end > length:
    parser.error(f'start/end ({args.start}/{args.end}) out of range for route length ({length}s)')
  return args

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

def main():
  p = ArgumentParser(prog='clip.py', description='clip your openpilot route.', epilog='comma.ai')
  if shutil.which('ffmpeg') is None:
    p.exit(1, 'clip.py: error: missing ffmpeg command, is it installed?\n')
  if platform.system() == 'Linux' and shutil.which('Xvfb') is None:
    p.exit(1, 'clip.py: error: missing Xvfb command, is it installed?\n')

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
  p.add_argument('-t', '--title', help='overlay this title on the video', type=validate_title)
  args = parse_args(p)

  try:
    clip(
      data_dir=args.data_dir, quality=args.quality, prefix=args.prefix, route=args.route,
      out=args.output, start=args.start, end=args.end, speed=args.speed,
      target_mb=args.file_size, title=args.title,
    )
  except (KeyboardInterrupt, Exception) as e:
    logger.exception('encountered error', exc_info=e)
    sys.exit(1)


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s\t%(message)s')
  main()