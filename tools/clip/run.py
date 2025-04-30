#!/usr/bin/env python3

import atexit
import logging
import os
import platform
import shutil
import time
from argparse import ArgumentParser, ArgumentTypeError
from collections.abc import Sequence
from pathlib import Path
from random import randint
from subprocess import Popen, PIPE
from typing import Literal

from cereal.messaging import SubMaster
from openpilot.common.basedir import BASEDIR
from openpilot.common.prefix import OpenpilotPrefix
from openpilot.tools.lib.route import SegmentRange, get_max_seg_number_cached

DEFAULT_OUTPUT = 'output.mp4'
DEMO_START = 90
DEMO_END = 105
DEMO_ROUTE = 'a2a0ccea32023010/2023-07-27--13-01-19'
FRAMERATE = 20
PIXEL_DEPTH = '24'
RESOLUTION = '2160x1080'
SECONDS_TO_WARM = 2
PROC_WAIT_SECONDS = 5

REPLAY = str(Path(BASEDIR, 'tools/replay/replay').resolve())
UI = str(Path(BASEDIR, 'selfdrive/ui/ui').resolve())

logger = logging.getLogger('clip.py')


def check_for_failure(proc: Popen):
  exit_code = proc.poll()
  if exit_code is not None and exit_code != 0:
    cmd = str(proc.args)
    if isinstance(proc.args, str):
      cmd = proc.args
    elif isinstance(proc.args, Sequence):
      cmd = str(proc.args[0])
    msg = f'{cmd} failed, exit code {exit_code}'
    logger.error(msg)
    stdout, stderr = proc.communicate()
    if stdout:
      logger.error(stdout.decode())
    if stderr:
      logger.error(stderr.decode())
    raise ChildProcessError(msg)


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

  # if using local files, don't worry about length check right now so we skip the network call
  # TODO: derive segment count from local FS
  if not args.data_dir:
    try:
      num_segs = get_max_seg_number_cached(SegmentRange(args.route))
    except Exception as e:
      parser.error(f'failed to get route length: {e}')

    # FIXME: length isn't exactly max segment seconds, simplify to replay exiting at end of data
    length = round(num_segs * 60)
    if args.start >= length:
      parser.error(f'start ({args.start}s) cannot be after end of route ({length}s)')
    if args.end > length:
      parser.error(f'end ({args.end}s) cannot be after end of route ({length}s)')

  return args


def start_proc(args: list[str], env: dict[str, str]):
  return Popen(args, env=env, stdout=PIPE, stderr=PIPE)


def validate_env(parser: ArgumentParser):
  if platform.system() not in ['Linux']:
    parser.exit(1, f'clip.py: error: {platform.system()} is not a supported operating system\n')
  for proc in ['Xvfb', 'ffmpeg']:
    if shutil.which(proc) is None:
      parser.exit(1, f'clip.py: error: missing {proc} command, is it installed?\n')
  for proc in [REPLAY, UI]:
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
  logger.info(f'clipping route {route}, start={start} end={end} quality={quality} target_filesize={target_size_mb}MB')

  begin_at = max(start - SECONDS_TO_WARM, 0)
  duration = end - start
  bit_rate_kbps = int(round(target_size_mb * 8 * 1024 * 1024 / duration / 1000))

  # TODO: evaluate creating fn that inspects /tmp/.X11-unix and creates unused display to avoid possibility of collision
  display = f':{randint(99, 999)}'

  ffmpeg_cmd = [
    'ffmpeg', '-y', '-video_size', RESOLUTION, '-framerate', str(FRAMERATE), '-f', 'x11grab', '-draw_mouse', '0',
    '-i', display, '-c:v', 'libx264', '-maxrate', f'{bit_rate_kbps}k', '-bufsize', f'{bit_rate_kbps*2}k', '-crf', '23',
    '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-f', 'mp4', '-t', str(duration), output_filepath,
  ]

  replay_cmd = [REPLAY, '-c', '1', '-s', str(begin_at), '--prefix', prefix]
  if data_dir:
    replay_cmd.extend(['--data_dir', data_dir])
  if quality == 'low':
    replay_cmd.append('--qcam')
  replay_cmd.append(route)

  ui_cmd = [UI, '-platform', 'xcb']
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
    procs = [replay_proc, ui_proc, xvfb_proc]

    logger.info('waiting for replay to begin (loading segments, may take a while)...')
    wait_for_frames(procs)

    logger.debug(f'letting UI warm up ({SECONDS_TO_WARM}s)...')
    time.sleep(SECONDS_TO_WARM)
    for proc in procs:
      check_for_failure(proc)

    ffmpeg_proc = start_proc(ffmpeg_cmd, env)
    procs.append(ffmpeg_proc)
    atexit.register(lambda: ffmpeg_proc.terminate())

    logger.info(f'recording in progress ({duration}s)...')
    ffmpeg_proc.wait(duration + PROC_WAIT_SECONDS)
    for proc in procs:
      check_for_failure(proc)
    logger.info(f'recording complete: {Path(output_filepath).resolve()}')


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
