#!/usr/bin/env python3

from argparse import ArgumentParser
from openpilot.common.prefix import OpenpilotPrefix
from openpilot.tools.clip.util import DEMO_ROUTE, DEMO_START, DEMO_END, monitor_processes, parse_args, validate_route, validate_env, wait_for_video
from pathlib import Path
from random import randint
import atexit
import logging
import os
import signal
import subprocess
import time

DEFAULT_OUTPUT = 'output.mp4'
FRAMERATE = 20
PIXEL_DEPTH = '24'
RESOLUTION = '2160x1080'
SECONDS_TO_WARM = 2

logger = logging.getLogger('clip.py')

def clip(data_dir: str | None, prefix: str, route: str, output_filepath: str, start_seconds: int, end_seconds: int, target_size_mb: int = 10):
  # offset the requested start point to allow the UI to "warm up" (sometimes the uiDebug msg has been seen but it has not drawn to the screen yet)
  begin_at = max(start_seconds - SECONDS_TO_WARM, 0)
  extra_buffer_seconds = min(SECONDS_TO_WARM, start_seconds)
  duration = end_seconds - start_seconds

  # TODO: evaluate creating fn that inspects /tmp/.X11-unix and creates unused display to avoid possibility of collision
  display_num = str(randint(99, 999))
  xauth = f'/tmp/.Xauthority-{prefix}-{display_num}'

  with OpenpilotPrefix(prefix, shared_download_cache=True) as _:
    env = os.environ.copy()
    env['XAUTHORITY'] = xauth
    env['QT_QPA_PLATFORM'] = 'xcb'

    ui_proc = subprocess.Popen(['xvfb-run', '-f', xauth, '-n', display_num, '-s', f'-screen 0 {RESOLUTION}x{PIXEL_DEPTH}', './selfdrive/ui/ui'], env=env)
    atexit.register(lambda: ui_proc.terminate())

    replay_args = ['./tools/replay/replay', '-c', '1', '-s', str(begin_at), '--no-loop', '--prefix', prefix]
    if data_dir:
      replay_args.extend(['--data_dir', data_dir])

    replay_proc = subprocess.Popen([*replay_args, route], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    atexit.register(lambda: replay_proc.terminate())

    logger.debug('waiting for replay to begin (loading segments, may take a while)...')
    wait_for_video(replay_proc)
    if extra_buffer_seconds > 0:
      logger.debug(f'letting UI warm up ({extra_buffer_seconds}s)...')
      time.sleep(extra_buffer_seconds)

    bit_rate_kbps = int(round(target_size_mb * 8 * 1_000_000 / duration / 1000))

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
      output_filepath,
    ]
    ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    atexit.register(lambda: ffmpeg_proc.terminate())

    logger.debug(f'recording in progress ({duration}s)...')
    monitor_processes([ui_proc, replay_proc, ffmpeg_proc], ['ui', 'replay', 'ffmpeg'])
    time.sleep(duration)

    ffmpeg_proc.send_signal(signal.SIGINT)
    ffmpeg_proc.wait(timeout=5)
    replay_proc.terminate()
    replay_proc.wait(timeout=5)
    ui_proc.terminate()
    ui_proc.wait(timeout=5)



def main():
  logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(name)s %(levelname)s\t%(message)s')

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

  validate_env(p)
  args = parse_args(p)

  try:
    logger.info(f'clipping route {args.route}, start={args.start} end={args.end}')
    clip(args.data_dir, args.prefix, args.route, args.output, args.start, args.end)
    logger.info(f'recording complete: {Path(args.output).resolve()}')
  except KeyboardInterrupt as e:
    logging.exception('interrupted by user', exc_info=e)
  except Exception as e:
    logging.exception('encountered error', exc_info=e)
  finally:
    atexit._run_exitfuncs()


if __name__ == '__main__':
  main()
