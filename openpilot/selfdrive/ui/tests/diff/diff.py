#!/usr/bin/env python3
import os
import sys
import subprocess
import webbrowser
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from openpilot.common.basedir import BASEDIR

DIFF_OUT_DIR = Path(BASEDIR) / "selfdrive" / "ui" / "tests" / "diff" / "report"
HTML_TEMPLATE_PATH = Path(__file__).with_name("diff_template.html")


def extract_framehashes(video_path: Path) -> list[str]:
  cmd = ['ffmpeg', '-nostdin', '-i', video_path, '-map', '0:v:0', '-vsync', '0', '-f', 'framehash', '-hash', 'md5', '-']
  result = subprocess.run(cmd, capture_output=True, text=True, check=True)
  hashes = []
  for line in result.stdout.splitlines():
    if not line or line.startswith('#'):
      continue
    parts = line.split(',')
    if len(parts) < 4:
      continue
    hashes.append(parts[-1].strip())
  return hashes


def get_video_frame_hashes(video1: Path, video2: Path) -> tuple[list[str], list[str]]:
  """Hash every frame of both videos in parallel and return the two hash lists."""
  with ThreadPoolExecutor(max_workers=2) as executor:
    print("Generating frame hashes for both videos...")
    future1 = executor.submit(extract_framehashes, video1)
    future2 = executor.submit(extract_framehashes, video2)
    hashes1 = future1.result()
    hashes2 = future2.result()
  return hashes1, hashes2


def create_diff_video(video1: Path, video2: Path, output: Path) -> None:
  """Create a diff video of two clips using ffmpeg blend filter with difference mode."""
  cmd = ['ffmpeg', '-nostdin', '-i', video1, '-i', video2, '-filter_complex', 'blend=all_mode=difference', '-vsync', '0', '-y', output]
  subprocess.run(cmd, capture_output=True, check=True)


def find_frame_differences(hashes1: list[str], hashes2: list[str]) -> list[int]:
  """Compare two lists of frame hashes and return the indices of different frames."""
  different_frames = []
  for i, (h1, h2) in enumerate(zip(hashes1, hashes2, strict=False)):
    if h1 != h2:
      different_frames.append(i)
  return different_frames


def generate_html_report(videos: tuple[Path, Path], basedir: str, different_frames: list[int], frame_counts: tuple[int, int], diff_video_name: str) -> str:
  total_frames = max(frame_counts)
  frame_delta = frame_counts[1] - frame_counts[0]
  different_total = len(different_frames) + abs(frame_delta)

  result_text = (
    f"✅ Videos are identical! ({total_frames} frames)"
    if different_total == 0
    else f"❌ Found {different_total} different frames out of {total_frames} total ({different_total / total_frames * 100:.1f}%)."
    + (f" Video {'2' if frame_delta > 0 else '1'} is longer by {abs(frame_delta)} frames." if frame_delta != 0 else "")
  )
  print(f"  Results: {result_text}")

  # Load HTML template and replace placeholders
  html = HTML_TEMPLATE_PATH.read_text()
  placeholders = {
    "VIDEO1_SRC": os.path.join(basedir, videos[0].name),
    "VIDEO2_SRC": os.path.join(basedir, videos[1].name),
    "DIFF_SRC": os.path.join(basedir, diff_video_name),
    "RESULT_TEXT": result_text,
  }
  for key, value in placeholders.items():
    html = html.replace(f"${key}", value)

  return html


def main():
  parser = argparse.ArgumentParser(description='Compare two videos and generate HTML diff report')
  parser.add_argument('video1', help='First video file')
  parser.add_argument('video2', help='Second video file')
  parser.add_argument('output', nargs='?', default='diff.html', help='Output HTML file (default: diff.html)')
  parser.add_argument("--basedir", type=str, help="Base path for files in HTML report", default="")
  parser.add_argument('--no-open', action='store_true', help='Do not open HTML report in browser')

  args = parser.parse_args()

  if not args.output.lower().endswith('.html'):
    args.output += '.html'

  video1, video2 = Path(args.video1), Path(args.video2)
  missing = [str(p) for p in (video1, video2) if not p.is_file()]
  if missing:
    parser.error(f"Video file(s) not found: {', '.join(missing)}")

  diff_video_name = f"{Path(args.output).stem}.mp4"  # diff video name derived from output HTML name

  os.makedirs(DIFF_OUT_DIR, exist_ok=True)

  print("=" * 60)
  print("UI VIDEO DIFF - HTML REPORT")
  print("=" * 60)
  print(f"Video 1:      {video1}")
  print(f"Video 2:      {video2}")
  print(f"HTML output:  {args.output}")
  print(f"Diff video:   {diff_video_name}")
  print()

  print("[1/4] Starting full video diff generation in background thread...")
  diff_executor = ThreadPoolExecutor(max_workers=1)
  diff_future = diff_executor.submit(create_diff_video, video1, video2, DIFF_OUT_DIR / diff_video_name)

  print("[2/4] Hashing frames...")
  hashes1, hashes2 = get_video_frame_hashes(video1, video2)
  frame_counts = (len(hashes1), len(hashes2))
  print(f"  Found {frame_counts[0]} frames in video 1 and {frame_counts[1]} frames in video 2.")

  print("[3/4] Finding different frames...")
  different_frames = find_frame_differences(hashes1, hashes2)
  print(f"  Found {len(different_frames)} different frames.")

  print("[4/4] Generating HTML report...")
  html = generate_html_report((video1, video2), args.basedir, different_frames, frame_counts, diff_video_name)

  output_path = DIFF_OUT_DIR / args.output
  with open(output_path, 'w') as f:
    f.write(html)

  print(f"  Report generated at: {output_path}")

  # Open in browser by default
  if not args.no_open:
    print(f"Opening {args.output} in browser...")
    webbrowser.open(f'file://{os.path.abspath(output_path)}')

  # Wait for diff video generation to finish before exiting
  if not diff_future.done():
    print("Waiting for diff video generation to finish...")
  diff_future.result()
  diff_executor.shutdown()

  extra_frames = abs(frame_counts[0] - frame_counts[1])
  return 0 if (len(different_frames) + extra_frames) == 0 else 1


if __name__ == "__main__":
  sys.exit(main())
