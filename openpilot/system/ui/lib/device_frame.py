"""Frame comma four screenshots and recordings using the comma.ai product render."""
import argparse
import subprocess
from pathlib import Path

FRAME = Path(__file__).resolve().parent.parent / "assets/comma_four.png"
# Match the screen overlay in commaai/website's comma four product page.
SCREEN_X, SCREEN_Y, SCREEN_WIDTH, SCREEN_HEIGHT = 390, 716, 676, 288


def frame_args(mode: str, flip: bool = False, fps: int = 60) -> list[str]:
  if mode not in ("full", "faded"):
    raise ValueError("device frame must be full or faded")
  frame = FRAME.with_stem("comma_four_faded") if mode == "faded" else FRAME
  x, y = (SCREEN_X - 220, SCREEN_Y - 420) if mode == "faded" else (SCREEN_X, SCREEN_Y)
  flip_filter = "vflip," if flip else ""
  graph = "".join((
    f"[0:v]{flip_filter}scale=w={SCREEN_WIDTH}:h={SCREEN_HEIGHT}:force_original_aspect_ratio=decrease,",
    f"pad=w={SCREEN_WIDTH}:h={SCREEN_HEIGHT}:x=(ow-iw)/2:y=(oh-ih)/2,setsar=sar=1[screen];",
    f"[1:v][screen]overlay=x={x}:y={y}:shortest=1,format=yuv420p[out]",
  ))
  return ['-loop', '1', '-framerate', str(fps), '-i', str(frame), '-filter_complex_threads', '1', '-filter_complex', graph, '-map', '[out]']


def check_ffmpeg(ffmpeg: str):
  filters = subprocess.check_output([ffmpeg, '-hide_banner', '-filters'], stderr=subprocess.STDOUT, text=True)
  if not all(name in filters.split() for name in ('overlay', 'setsar', 'pad', 'scale')):
    raise ValueError("Device frames require a full FFmpeg build; set RECORD_FFMPEG=/usr/bin/ffmpeg or pass --ffmpeg /usr/bin/ffmpeg")


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('input', type=Path)
  parser.add_argument('output', type=Path)
  parser.add_argument('--ffmpeg', default='ffmpeg', help='full FFmpeg build (the bundled minimal build is unsupported)')
  parser.add_argument('--fps', type=int, default=30, help='output frame rate for video')
  parser.add_argument('--mode', choices=('full', 'faded'), default='full')
  args = parser.parse_args()
  check_ffmpeg(args.ffmpeg)
  image = args.output.suffix.lower() == '.png'
  output_args = ['-frames:v', '1', '-update', '1'] if image else ['-c:v', 'libx264', '-crf', '18', '-preset', 'veryfast']
  subprocess.run([args.ffmpeg, '-v', 'warning', '-y', '-i', str(args.input), *frame_args(args.mode, fps=args.fps), *output_args, str(args.output)], check=True)


if __name__ == '__main__':
  main()
