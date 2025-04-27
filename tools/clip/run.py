#!/usr/bin/env python3

from argparse import ArgumentParser, ArgumentTypeError
from cereal.messaging import SubMaster
from openpilot.common.prefix import OpenpilotPrefix
from subprocess import DEVNULL
from random import randint
import atexit
import os
import signal
import subprocess
import time


RESOLUTION = '2160x1080'
PIXEL_DEPTH = '24'
FRAMERATE = 20
DEFAULT_OUTPUT = 'output.mp4'
DEMO_START = 20
DEMO_END = 30
DEMO_ROUTE = 'a2a0ccea32023010/2023-07-27--13-01-19'


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

  # TODO: need to wait a little longer, sometimes UI doesn't react fast enough
  time.sleep(0.75)


def main(data_dir: str | None, route: str, output_filepath: str, start_seconds: int, end_seconds: int):
  # TODO: evaluate creating fn that inspects /tmp/.X11-unix and creates unused display to avoid possibility of collision
  display_num = str(randint(99, 999))

  duration = end_seconds - start_seconds

  env = os.environ.copy()
  xauth = f'/tmp/clip-xauth--{display_num}'
  env['XAUTHORITY'] = xauth
  env['QT_QPA_PLATFORM'] = 'xcb'

  ui_proc = subprocess.Popen(['xvfb-run', '-f', xauth, '-n', display_num, '-s', f'-screen 0 {RESOLUTION}x{PIXEL_DEPTH}', './selfdrive/ui/ui'], env=env)
  atexit.register(lambda: ui_proc.terminate())

  replay_args = ['./tools/replay/replay', '-c', '1', '-s', str(start_seconds), '--no-loop', '--prefix', str(env.get('OPENPILOT_PREFIX'))]
  if data_dir:
    replay_args.extend(['--data_dir', data_dir])
  replay_proc = subprocess.Popen([*replay_args, route], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  atexit.register(lambda: replay_proc.terminate())

  print('waiting for replay to begin (may take a while)...')
  wait_for_video(replay_proc)

  ffmpeg_cmd = [
    'ffmpeg',
    '-y',
    '-video_size', RESOLUTION,
    '-framerate', str(FRAMERATE),
    '-f',
    'x11grab',
    '-draw_mouse',
    '0',
    '-i', f':{display_num}',
    '-c:v',
    'libx264',
    '-preset',
    'ultrafast',
    '-pix_fmt',
    'yuv420p',
    output_filepath,
  ]
  ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, env=env, stdout=DEVNULL, stderr=DEVNULL)
  atexit.register(lambda: ffmpeg_proc.terminate())

  print('recording in progress...')
  time.sleep(duration)

  ffmpeg_proc.send_signal(signal.SIGINT)
  ffmpeg_proc.wait(timeout=5)
  ui_proc.terminate()
  ui_proc.wait(timeout=5)

  print(f'recording complete: {output_filepath}')


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

  if args.end is not None and args.start is not None and args.end <= args.start:
    parser.error(f'end ({args.end}) must be greater than start ({args.start})')

  return args


def validate_route(route: str):
  if route.count('/') not in (1, 3):
    raise ArgumentTypeError('route must include or exclude timing, example: ' + DEMO_ROUTE)
  return route


if __name__ == '__main__':
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
  p.add_argument('-s', '--start', help='Start clipping at <start> seconds', type=int)
  p.add_argument('-e', '--end', help='Stop clipping at <end> seconds', type=int)

  args = parse_args(p)

  try:
    with OpenpilotPrefix(args.prefix, shared_download_cache=True) as p:
      print(f'clipping route {args.route}, start={args.start} end={args.end}')
      main(args.data_dir, args.route, args.output, args.start, args.end)
  except KeyboardInterrupt:
    print('Interrupted by user')
  except Exception as e:
    print(f'Error: {e}')
  finally:
    atexit._run_exitfuncs()
