from argparse import ArgumentParser
from msgq.visionipc import VisionIpcClient, VisionStreamType
from cereal.messaging import SubMaster
import os
import signal
import subprocess
from subprocess import DEVNULL
import time
import atexit
from random import randint

DEFAULT_DISPLAY = ":99"
RESOLUTION = "2160x1080"
PIXEL_DEPTH = "24"
FRAMERATE = 20
DEFAULT_OUTPUT = "output.mp4"
DEMO_ROUTE = "a2a0ccea32023010/2023-07-27--13-01-19/0"


def wait_for_video():
  sm = SubMaster(['uiDebug'])
  no_frames_drawn = True
  while no_frames_drawn:
    sm.update()
    no_frames_drawn = sm['uiDebug'].drawTimeMillis == 0.


def ensure_xvfb(display: str):
  xvfb_cmd = ["Xvfb", display, "-screen", "0", f"{RESOLUTION}x{PIXEL_DEPTH}"]
  xvfb_proc = subprocess.Popen(xvfb_cmd, stdout=DEVNULL, stderr=DEVNULL)
  time.sleep(1)
  if xvfb_proc.poll() is not None:
    raise RuntimeError(f"Failed to start Xvfb on display {display}")
  return xvfb_proc


def main(route: str, output_filepath: str, start_seconds: int, end_seconds: int):
  assert end_seconds > start_seconds, 'end must be greater than start'
  display = ':' + str(randint(99, 999))

  duration = end_seconds - start_seconds

  env = os.environ.copy()
  env["DISPLAY"] = display
  env["QT_QPA_PLATFORM"] = "xcb"

  print(f'starting xvfb on display {display}')
  xvfb_proc = ensure_xvfb(display)
  atexit.register(lambda: xvfb_proc.terminate())  # Ensure cleanup on exit

  ui_args = ["./selfdrive/ui/ui"]
  ui_proc = subprocess.Popen(ui_args, env=env, stdout=DEVNULL, stderr=DEVNULL)
  atexit.register(lambda: ui_proc.terminate())

  replay_proc = subprocess.Popen(["./tools/replay/replay", "-c", "1", "-s", str(start_seconds), "--no-loop", "--demo"], env=env, stdout=DEVNULL, stderr=DEVNULL)
  atexit.register(lambda: replay_proc.terminate())

  # Wait for video data
  wait_for_video()
  time.sleep(2)

  # Start FFmpeg
  ffmpeg_cmd = [
    "ffmpeg",
    "-y",
    "-video_size",
    RESOLUTION,
    "-framerate",
    str(FRAMERATE),
    "-f",
    "x11grab",
    "-draw_mouse",
    "0",
    "-i",
    env.get('DISPLAY'),
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

  print(f'starting at {start_seconds} seconds and clipping {duration} seconds')
  time.sleep(duration)

  # Stop FFmpeg gracefully
  ffmpeg_proc.send_signal(signal.SIGINT)
  ffmpeg_proc.wait(timeout=5)

  # Clean up
  ui_proc.terminate()
  ui_proc.wait(timeout=5)
  xvfb_proc.terminate()
  xvfb_proc.wait(timeout=5)

  print(f"Recording complete: {output_filepath}")


if __name__ == "__main__":
  p = ArgumentParser(
    prog='clip.py',
    description='Clip your openpilot route.',
    epilog='comma.ai'
  )
  p.add_argument('-r', '--route', help='Route', default=DEMO_ROUTE)
  p.add_argument('-o', '--output', help='Output clip to (.mp4)', default=DEFAULT_OUTPUT)
  p.add_argument('-s', '--start', help='Start clipping at <start> seconds', type=int, required=True)
  p.add_argument('-e', '--end', help='Stop clipping at <end> seconds', type=int, required=True)
  args = p.parse_args()
  try:
    main(args.route, args.output, args.start, args.end)
  except KeyboardInterrupt:
    print("Interrupted by user")
  except Exception as e:
    print(f"Error: {e}")
  finally:
    # Ensure all processes are terminated
    atexit._run_exitfuncs()
