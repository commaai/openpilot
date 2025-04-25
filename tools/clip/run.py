from argparse import ArgumentParser
from msgq.visionipc import VisionIpcClient, VisionStreamType
import os
import signal
import subprocess
from subprocess import DEVNULL
import time
import atexit

XVFB_DISPLAY = ":99"
RESOLUTION = "2160x1080"
PIXEL_DEPTH = "24"
FRAMERATE = 20
DEFAULT_OUTPUT = "output.mp4"
DEMO_ROUTE = "a2a0ccea32023010/2023-07-27--13-01-19/0"


def wait_for_video():
  vipc = VisionIpcClient('camerad', VisionStreamType.VISION_STREAM_ROAD, True)
  while True:
    if not vipc.is_connected():
      vipc.connect(True)
    new_data = vipc.recv()
    if new_data is not None and new_data.data.any():
      return
    time.sleep(0.01)  # Prevent tight loop


def ensure_xvfb():
  """Start Xvfb and return the process."""
  xvfb_cmd = ["Xvfb", XVFB_DISPLAY, "-screen", "0", f"{RESOLUTION}x{PIXEL_DEPTH}"]
  xvfb_proc = subprocess.Popen(xvfb_cmd, stdout=DEVNULL, stderr=DEVNULL)
  time.sleep(1)  # Give Xvfb time to start
  if xvfb_proc.poll() is not None:
    raise RuntimeError("Failed to start Xvfb")
  return xvfb_proc


def main(route: str, output_filepath: str, start_seconds: int, end_seconds: int):
  assert end_seconds > start_seconds, 'end must be greater than start'

  duration = end_seconds - start_seconds
  # Set up environment
  env = os.environ.copy()
  env["DISPLAY"] = XVFB_DISPLAY  # Set DISPLAY for all processes
  env["QT_QPA_PLATFORM"] = "xcb"

  # Start Xvfb
  xvfb_proc = ensure_xvfb()
  atexit.register(lambda: xvfb_proc.terminate())  # Ensure cleanup on exit

  # Start UI process
  ui_args = ["./selfdrive/ui/ui"]
  ui_proc = subprocess.Popen(ui_args, env=env, stdout=DEVNULL, stderr=DEVNULL)
  atexit.register(lambda: ui_proc.terminate())

  # Start replay process
  replay_proc = subprocess.Popen(["./tools/replay/replay", "-s", str(start_seconds), "--no-loop", route], env=env, stdout=DEVNULL, stderr=DEVNULL)
  atexit.register(lambda: replay_proc.terminate())

  # Wait for video data
  wait_for_video()
  time.sleep(1)

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
    XVFB_DISPLAY,
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
