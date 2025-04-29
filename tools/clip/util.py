from argparse import ArgumentParser, ArgumentTypeError
from cereal.messaging import SubMaster
import platform
import shutil
import subprocess
import time

DEMO_START = 20
DEMO_END = 30
DEMO_ROUTE = 'a2a0ccea32023010/2023-07-27--13-01-19'

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

  # TODO: need to wait a little longer, sometimes UI doesn't react fast enough
  time.sleep(0.75)
