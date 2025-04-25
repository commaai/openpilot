from argparse import ArgumentParser
from cereal.messaging import SubMaster
import os
import signal
import subprocess
from subprocess import DEVNULL
import time
import atexit
from random import randint

from openpilot.common.prefix import OpenpilotPrefix

DEFAULT_DISPLAY = ":99"
RESOLUTION = "2160x1080"
PIXEL_DEPTH = "24"
FRAMERATE = 20
DEFAULT_OUTPUT = "output.mp4"
DEMO_START = 20
DEMO_END = 30
DEMO_ROUTE = "a2a0ccea32023010/2023-07-27--13-01-19"


def wait_for_video():
  sm = SubMaster(['uiDebug'])
  no_frames_drawn = True
  while no_frames_drawn:
    sm.update()
    no_frames_drawn = sm['uiDebug'].drawTimeMillis == 0.


def main(route: str, output_filepath: str, start_seconds: int, end_seconds: int):
  # TODO: evaluate creating fn that inspects /tmp/.X11-unix and creates unused display to avoid possibility of collision
  display_num = str(randint(99, 999))

  duration = end_seconds - start_seconds

  env = os.environ.copy()
  env["QT_QPA_PLATFORM"] = "xcb"

  ui_proc = subprocess.Popen(['xvfb-run', '-n', display_num, '-s', f'-screen 0 {RESOLUTION}x{PIXEL_DEPTH}', './selfdrive/ui/ui'], env=env)
  atexit.register(lambda: ui_proc.terminate())

  replay_proc = subprocess.Popen(
    ["./tools/replay/replay", "-c", "1", "-s", str(start_seconds), "--no-loop", "--prefix", env.get('OPENPILOT_PREFIX'), route],
    env=env, stdout=DEVNULL, stderr=DEVNULL)
  atexit.register(lambda: replay_proc.terminate())

  wait_for_video()

  ffmpeg_cmd = [
    "ffmpeg",
    "-y",
    "-video_size", RESOLUTION,
    "-framerate", str(FRAMERATE),
    "-f",
    "x11grab",
    "-draw_mouse",
    "0",
    "-i", ":" + display_num,
    "-c:v",
    "libx264",
    "-preset",
    "ultrafast",
    "-pix_fmt",
    "yuv420p",
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

  print(f"recording complete: {output_filepath}")


if __name__ == "__main__":
  p = ArgumentParser(
    prog='clip.py',
    description='Clip your openpilot route.',
    epilog='comma.ai'
  )
  p.add_argument('-p', '--prefix', help='openpilot prefix', default=f'clip_{randint(100, 99999)}')
  p.add_argument('-r', '--route', help='Route', default=DEMO_ROUTE)
  p.add_argument('-o', '--output', help='Output clip to (.mp4)', default=DEFAULT_OUTPUT)
  p.add_argument('-s', '--start', help='Start clipping at <start> seconds', type=int, default=DEMO_START)
  p.add_argument('-e', '--end', help='Stop clipping at <end> seconds', type=int, default=DEMO_END)
  args = p.parse_args()
  assert args.end > args.start, 'end must be greater than start'
  assert args.route.count('/') == 1 or args.route.count('/') == 3, 'route must include or exclude timing, example: ' + DEMO_ROUTE
  if args.route.count('/') == 3:
    parts = args.route.split('/')
    args.start = int(parts[2])
    args.end = int(parts[3])
    args.route = '/'.join(parts[:2])
  print(f'clipping route {args.route}, start={args.start} end={args.end}')
  try:
    with OpenpilotPrefix(args.prefix, shared_download_cache=True) as p:
      main(args.route, args.output, args.start, args.end)
  except KeyboardInterrupt:
    print("Interrupted by user")
  except Exception as e:
    print(f"Error: {e}")
  finally:
    atexit._run_exitfuncs()
