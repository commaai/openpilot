#!/usr/bin/env python3

from argparse import ArgumentParser, ArgumentTypeError
from cereal.messaging import SubMaster
from openpilot.common.api import api_get
from openpilot.common.prefix import OpenpilotPrefix
from pathlib import Path
from random import randint
import atexit
import logging
import os
import platform
import shutil
import subprocess
import time

DEFAULT_OUTPUT = 'output.mp4'
DEMO_START = 20
DEMO_END = 30
DEMO_ROUTE = 'a2a0ccea32023010/2023-07-27--13-01-19'
FRAMERATE = 20
PIXEL_DEPTH = '24'
RESOLUTION = '2160x1080'
SECONDS_TO_WARM = 2

logger = logging.getLogger('clip.py')


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
    raise ValueError('route not found')
  if resp.status_code == 403:
    raise ValueError('route not public')
  if resp.status_code != 200:
    raise ValueError('unknown route request error code: ' + resp.status_code)
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


def validate_env(parser: ArgumentParser):
  if platform.system() not in ['Linux']:
    parser.exit(1, f'clip.py: error: {platform.system()} is not a supported operating system\n')
  for proc in ['xvfb-run', 'ffmpeg']:
    if shutil.which(proc) is None:
      parser.exit(1, f'clip.py: error: missing {proc} command, is it installed?\n')
  for proc in ['selfdrive/ui/ui', 'tools/replay/replay']:
    if shutil.which(proc) is None:
      parser.exit(1, f'clip.py: error: missing {proc} command, did you build openpilot yet?\n')


def validate_route(route: str):
  if route.count('/') not in (1, 3):
    raise ArgumentTypeError('route must include or exclude timing, example: ' + DEMO_ROUTE)
  return route


def wait_for_video(proc: subprocess.Popen):
  sm = SubMaster(['uiDebug'])
  no_frames_drawn = True
  while no_frames_drawn:
    sm.update()
    no_frames_drawn = sm['uiDebug'].drawTimeMillis == 0.
    if proc.poll() is not None:
      stdout, stderr = proc.communicate()
      print('-' * 16, ' replay output ', '-' * 16)
      print(stdout.decode().strip(), stderr.decode().strip())
      print('-' * 49)
      raise RuntimeError('replay failed to start!')


def clip(data_dir: str | None, low_quality: bool, prefix: str, route: str, output_filepath: str, start_seconds: int, end_seconds: int, target_size_mb: int = 10):
  route_dict = get_route(route)
  begin_at, start_seconds, end_seconds, duration = clip_timing(route_dict, start_seconds, end_seconds)
  logger.info(f'clipping route {route}, start={start_seconds} end={end_seconds}')

  # TODO: evaluate creating fn that inspects /tmp/.X11-unix and creates unused display to avoid possibility of collision
  display_num = str(randint(99, 999))

  replay_args = ['./tools/replay/replay', '--all', '-c', '1', '-s', str(begin_at), '--no-loop', '--prefix', prefix]
  if data_dir:
    replay_args.extend(['--data_dir', data_dir])
  if low_quality:
    replay_args.append('--qcam')

  bit_rate_kbps = int(round(target_size_mb * 8 * 1024 * 1024 / duration / 1000))

  ffmpeg_cmd = [
    'ffmpeg',
    '-y',
    '-video_size', RESOLUTION,
    '-framerate', str(FRAMERATE),
    '-f', 'x11grab',
    '-draw_mouse', '0',
    '-i', f':{display_num}',
    '-c:v', 'libx264',
    '-crf', '23',
    '-maxrate', f'{bit_rate_kbps}k',
    '-bufsize', f'{bit_rate_kbps * 2}k',
    '-preset', 'ultrafast',
    '-pix_fmt', 'yuv420p',
    '-movflags', '+faststart',
    '-t', str(duration),
    output_filepath,
  ]

  with OpenpilotPrefix(prefix, shared_download_cache=True) as _:
    env = os.environ.copy()
    env['DISPLAY'] = ':' + display_num

    xvfb_cmd = ['Xvfb', env['DISPLAY'], '-terminate', '-screen', '0', f'{RESOLUTION}x{PIXEL_DEPTH}']

    xvfb_proc = subprocess.Popen(xvfb_cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    atexit.register(lambda: xvfb_proc.terminate())

    ui_proc = subprocess.Popen(['./selfdrive/ui/ui', '-platform', 'xcb'], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    atexit.register(lambda: ui_proc.terminate())

    replay_proc = subprocess.Popen([*replay_args, route], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    atexit.register(lambda: replay_proc.terminate())

    logger.info('waiting for replay to begin (loading segments, may take a while)...')
    wait_for_video(replay_proc)

    logger.debug(f'letting UI warm up ({SECONDS_TO_WARM}s)...')
    time.sleep(SECONDS_TO_WARM)

    ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    atexit.register(lambda: ffmpeg_proc.terminate())

    logger.info(f'recording in progress ({duration}s)...')
    ffmpeg_proc.wait(duration + 5)
    logger.info(f'recording complete: {Path(output_filepath).resolve()}')


def main():
  logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s\t%(message)s')

  p = ArgumentParser(
    prog='clip.py',
    description='Clip your openpilot route.',
    epilog='comma.ai'
  )
  route_group = p.add_mutually_exclusive_group(required=True)
  route_group.add_argument('route', nargs='?', type=validate_route,
    help=f'The route (e.g. {DEMO_ROUTE} or {DEMO_ROUTE}/{DEMO_START}/{DEMO_END})')
  route_group.add_argument('--demo', help='Use the demo route', action='store_true')
  p.add_argument('-p', '--prefix', help='openpilot prefix', default=f'clip_{randint(100, 99999)}')
  p.add_argument('-o', '--output', help='Output clip to (.mp4)', default=DEFAULT_OUTPUT)
  p.add_argument('-dir', '--data_dir', help='Local directory where route data is stored')
  p.add_argument('-l', '--low-quality', help='use qcams for video (lower quality)', action='store_true')
  p.add_argument('-s', '--start', help='Start clipping at <start> seconds', type=int)
  p.add_argument('-e', '--end', help='Stop clipping at <end> seconds', type=int)

  validate_env(p)
  args = parse_args(p)

  try:
    clip(args.data_dir, args.low_quality, args.prefix, args.route, args.output, args.start, args.end)
  except KeyboardInterrupt as e:
    logging.exception('interrupted by user', exc_info=e)
  except Exception as e:
    logging.exception('encountered error', exc_info=e)
  finally:
    atexit._run_exitfuncs()


if __name__ == '__main__':
  main()
