#!/usr/bin/env python3

from argparse import ArgumentParser, ArgumentTypeError
from cereal.messaging import SubMaster
from collections.abc import Sequence
from openpilot.common.api import api_get
from openpilot.common.prefix import OpenpilotPrefix
from pathlib import Path
from random import randint
from subprocess import Popen, PIPE
from typing import Literal
import atexit
import logging
import os
import platform
import shutil
import time

DEFAULT_OUTPUT = 'output.mp4'
DEMO_START = 20
DEMO_END = 30
DEMO_ROUTE = 'a2a0ccea32023010/2023-07-27--13-01-19'
FRAMERATE = 20
PIXEL_DEPTH = '24'
RESOLUTION = '2160x1080'
SECONDS_TO_WARM = 2
PROC_WAIT_SECONDS = 5

logger = logging.getLogger('clip.py')


def check_for_failure(proc: Popen):
  exit_code = proc.poll()
  if exit_code is not None and exit_code != 0:
    cmd = str(proc.args)
    if isinstance(proc.args, str):
      cmd = proc.args
    elif isinstance(proc.args, Sequence):
      cmd = str(proc.args[0])

    msg = f'{cmd} failed, exit code {proc.poll()}'
    logger.error(msg)
    stdout, stderr = proc.communicate()
    if stdout:
      logger.error(stdout.decode())
    if stderr:
      logger.error(stderr.decode())
    raise ChildProcessError(msg)


def clip_timing(route_dict: dict, start_seconds: int, end_seconds: int):
  length = round((route_dict['maxqlog'] + 1) * 60)
  begin_at = max(start_seconds - SECONDS_TO_WARM, 0)
  end_seconds = min(end_seconds, length)
  duration = end_seconds - start_seconds

  # FIXME: length isn't exactly max segment seconds, replay should exit at end of route
  assert start_seconds < length, f'start ({start_seconds}s) cannot be after end of route ({length}s)'
  assert start_seconds < end_seconds, f'end ({end_seconds}s) cannot be after start ({start_seconds}s)'

  return begin_at, start_seconds, end_seconds, duration


def get_route(route: str):
  dongle, route_id = route.split('/')
  resp = api_get(f'/v1/route/{dongle}|{route_id}')
  if resp.status_code == 404:
    raise ValueError(f'route {route} not found')
  if resp.status_code == 403:
    raise PermissionError(f'route {route} not public')
  if resp.status_code != 200:
    raise RuntimeError(f'unknown request error code for route {route}: {resp.status_code}')
  return resp.json()


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
  if args.start < SECONDS_TO_WARM:
    parser.error(f'start must be greater than {SECONDS_TO_WARM}s to allow the UI time to warm up')

  return args


def start_proc(args: list[str], env: dict[str, str]):
  return Popen(args, env=env, stdout=PIPE, stderr=PIPE)


def validate_env(parser: ArgumentParser):
  if platform.system() not in ['Linux']:
    parser.exit(1, f'clip.py: error: {platform.system()} is not a supported operating system\n')
  for proc in ['xvfb-run', 'ffmpeg']:
    if shutil.which(proc) is None:
      parser.exit(1, f'clip.py: error: missing {proc} command, is it installed?\n')
  for proc in ['selfdrive/ui/ui', 'tools/replay/replay']:
    if shutil.which(proc) is None:
      parser.exit(1, f'clip.py: error: missing {proc} command, did you build openpilot yet?\n')


def validate_output_file(output_file: str):
  if not output_file.endswith('.mp4'):
    raise ArgumentTypeError('output must be an mp4')
  return output_file


def validate_route(route: str):
  if route.count('/') not in (1, 3):
    raise ArgumentTypeError(f'route must include or exclude timing, example: {DEMO_ROUTE}')
  return route


def wait_for_frames(procs: list[Popen]):
  sm = SubMaster(['uiDebug'])
  no_frames_drawn = True
  while no_frames_drawn:
    sm.update()
    no_frames_drawn = sm['uiDebug'].drawTimeMillis == 0.
    for proc in procs:
      check_for_failure(proc)


def clip(data_dir: str | None, quality: Literal['low', 'high'], prefix: str, route: str, output_filepath: str, start: int, end: int, target_size_mb: int):
  route_dict = get_route(route)
  begin_at, start, end, duration = clip_timing(route_dict, start, end)
  bit_rate_kbps = int(round(target_size_mb * 8 * 1024 * 1024 / duration / 1000))

  logger.info(f'clipping route {route}, start={start} end={end} quality={quality} target_filesize={target_size_mb}MB')

  # TODO: evaluate creating fn that inspects /tmp/.X11-unix and creates unused display to avoid possibility of collision
  display = f':{randint(99, 999)}'

  ffmpeg_cmd = [
    'ffmpeg', '-y', '-video_size', RESOLUTION, '-framerate', str(FRAMERATE), '-f', 'x11grab', '-draw_mouse', '0',
    '-i', display, '-c:v', 'libx264', '-crf', '23', '-maxrate', f'{bit_rate_kbps}k', '-bufsize', f'{bit_rate_kbps * 2}k',
    '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-f', 'MP4', '-t', str(duration), output_filepath,
  ]

  replay_cmd = ['./tools/replay/replay', '-c', '1', '-s', str(begin_at), '--prefix', prefix]
  if data_dir:
    replay_cmd.extend(['--data_dir', data_dir])
  if quality == 'low':
    replay_cmd.append('--qcam')
  replay_cmd.append(route)

  ui_cmd = ['./selfdrive/ui/ui', '-platform', 'xcb']
  xvfb_cmd = ['Xvfb', display, '-terminate', '-screen', '0', f'{RESOLUTION}x{PIXEL_DEPTH}']

  with OpenpilotPrefix(prefix, shared_download_cache=True):
    env = os.environ.copy()
    env['DISPLAY'] = display

    xvfb_proc = start_proc(xvfb_cmd, env)
    atexit.register(lambda: xvfb_proc.terminate())
    ui_proc = start_proc(ui_cmd, env)
    atexit.register(lambda: ui_proc.terminate())
    replay_proc = start_proc(replay_cmd, env)
    atexit.register(lambda: replay_proc.terminate())

    logger.info('waiting for replay to begin (loading segments, may take a while)...')
    wait_for_frames([replay_proc, ui_proc])

    logger.debug(f'letting UI warm up ({SECONDS_TO_WARM}s)...')
    time.sleep(SECONDS_TO_WARM)
    check_for_failure(replay_proc)
    check_for_failure(ui_proc)

    ffmpeg_proc = start_proc(ffmpeg_cmd, env)
    atexit.register(lambda: ffmpeg_proc.terminate())

    logger.info(f'recording in progress ({duration}s)...')
    ffmpeg_proc.wait(duration + PROC_WAIT_SECONDS)
    ffmpeg_exit_code = ffmpeg_proc.poll()
    if ffmpeg_exit_code is not None and ffmpeg_exit_code != 0:
      check_for_failure(ffmpeg_proc)
    else:
      logger.info(f'recording complete: {Path(output_filepath).resolve()}')


def main():
  p = ArgumentParser(prog='clip.py', description='clip your openpilot route.', epilog='comma.ai')
  validate_env(p)
  route_group = p.add_mutually_exclusive_group(required=True)
  route_group.add_argument('route', nargs='?', type=validate_route, help=f'The route (e.g. {DEMO_ROUTE} or {DEMO_ROUTE}/{DEMO_START}/{DEMO_END})')
  route_group.add_argument('--demo', help='use the demo route', action='store_true')
  p.add_argument('-d', '--data-dir', help='local directory where route data is stored')
  p.add_argument('-e', '--end', help='stop clipping at <end> seconds', type=int)
  p.add_argument('-f', '--file-size', help='target file size (Discord/GitHub support max 10MB, default is 9.5MB)', type=float, default=9.5)
  p.add_argument('-o', '--output', help='output clip to (.mp4)', type=validate_output_file, default=DEFAULT_OUTPUT)
  p.add_argument('-p', '--prefix', help='openpilot prefix', default=f'clip_{randint(100, 99999)}')
  p.add_argument('-q', '--quality', help='quality of camera (low = qcam, high = hevc)', choices=['low', 'high'], default='high')
  p.add_argument('-s', '--start', help='start clipping at <start> seconds', type=int)
  args = parse_args(p)
  try:
    clip(args.data_dir, args.quality, args.prefix, args.route, args.output, args.start, args.end, args.file_size)
  except KeyboardInterrupt as e:
    logger.exception('interrupted by user', exc_info=e)
  except Exception as e:
    logger.exception('encountered error', exc_info=e)
  finally:
    atexit._run_exitfuncs()


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s\t%(message)s')
  main()
