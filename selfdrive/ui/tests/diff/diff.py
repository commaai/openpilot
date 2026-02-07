#!/usr/bin/env python3
import os
import sys
import subprocess
import tempfile
import base64
import webbrowser
import argparse
from pathlib import Path
from openpilot.common.basedir import BASEDIR

DIFF_OUT_DIR = Path(BASEDIR) / "selfdrive" / "ui" / "tests" / "diff" / "report"


def extract_frames(video_path, output_dir):
  output_pattern = str(output_dir / "frame_%04d.png")
  cmd = ['ffmpeg', '-i', video_path, '-vsync', '0', output_pattern, '-y']
  subprocess.run(cmd, capture_output=True, check=True)
  frames = sorted(output_dir.glob("frame_*.png"))
  return frames


def compare_frames(frame1_path, frame2_path):
  result = subprocess.run(['cmp', '-s', frame1_path, frame2_path])
  return result.returncode == 0


def frame_to_data_url(frame_path):
  with open(frame_path, 'rb') as f:
    data = f.read()
  return f"data:image/png;base64,{base64.b64encode(data).decode()}"


def create_diff_video(video1, video2, output_path):
  """Create a diff video using ffmpeg blend filter with difference mode."""
  print("Creating diff video...")
  cmd = ['ffmpeg', '-i', video1, '-i', video2, '-filter_complex', '[0:v]blend=all_mode=difference', '-vsync', '0', '-y', output_path]
  subprocess.run(cmd, capture_output=True, check=True)


def find_differences(video1, video2) -> tuple[list[int], list, tuple[int, int]]:
  with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir = Path(tmpdir)

    print(f"Extracting frames from {video1}...")
    frames1_dir = tmpdir / "frames1"
    frames1_dir.mkdir()
    frames1 = extract_frames(video1, frames1_dir)

    print(f"Extracting frames from {video2}...")
    frames2_dir = tmpdir / "frames2"
    frames2_dir.mkdir()
    frames2 = extract_frames(video2, frames2_dir)

    print(f"Comparing {len(frames1)} frames...")
    different_frames: list[int] = []
    frame_data = []

    for i, (f1, f2) in enumerate(zip(frames1, frames2, strict=False)):
      is_different = not compare_frames(f1, f2)
      if is_different:
        different_frames.append(i)

      if i < 10 or i >= len(frames1) - 10 or is_different:
        frame_data.append({'index': i, 'different': is_different, 'frame1_url': frame_to_data_url(f1), 'frame2_url': frame_to_data_url(f2)})

    return different_frames, frame_data, (len(frames1), len(frames2))


def generate_html_report(videos: tuple[str, str], basedir: str, different_frames: list[int], frame_data, frame_counts: tuple[int, int]):
  chunks = []
  if different_frames:
    current_chunk = [different_frames[0]]
    for i in range(1, len(different_frames)):
      if different_frames[i] == different_frames[i - 1] + 1:
        current_chunk.append(different_frames[i])
      else:
        chunks.append(current_chunk)
        current_chunk = [different_frames[i]]
    chunks.append(current_chunk)

  total_frames = max(frame_counts)
  frame_delta = frame_counts[1] - frame_counts[0]
  different_total = len(different_frames) + abs(frame_delta)

  result_text = (
    f"✅ Videos are identical! ({total_frames} frames)"
    if different_total == 0
    else f"❌ Found {different_total} different frames out of {total_frames} total ({different_total / total_frames * 100:.1f}%)."
    + (f" Video {'2' if frame_delta > 0 else '1'} is longer by {abs(frame_delta)} frames." if frame_delta != 0 else "")
  )

  def render_video_cell(video_id: str, title: str, path: str, is_diff=False):
    return f"""
<td width='33%'>
  <p><strong>{title}</strong></p>
  <video id='{video_id}' width='100%' autoplay muted {'' if is_diff else "onplay='syncVideos()'"}>
    <source src='{os.path.join(basedir, os.path.basename(path))}' type='video/mp4'>
    Your browser does not support the video tag.
  </video>
</td>
"""

  html = f"""<h2>UI Diff</h2>
<table>
<tr>
{render_video_cell("video1", "Video 1", videos[0])}
{render_video_cell("video2", "Video 2", videos[1])}
{render_video_cell("diffVideo", "Pixel Diff", 'diff.mp4', is_diff=True)}
</tr>
</table>
<script>
const videos = [
  document.getElementById('video1'),
  document.getElementById('video2'),
  document.getElementById('diffVideo'),
];

const isEnded = (v) => v.ended || (Number.isFinite(v.duration) && v.currentTime >= (v.duration - 0.05));
const playAll = () => videos.forEach((v) => v.play());

function syncVideos() {{
  const t = Math.min(...videos.map((v) => v.currentTime));
  videos.forEach((v) => {{ v.currentTime = t; }});
  playAll();
}}

function handleEnded(endedVideo) {{
  endedVideo.pause();
  if (videos.every(isEnded)) {{
    videos.forEach((v) => {{ v.currentTime = 0; }});
    playAll();
  }}
}}

videos.forEach((v) => {{
  v.addEventListener('timeupdate', () => {{
    videos.forEach((other) => {{
      if (other !== v && !isEnded(other) && Math.abs(v.currentTime - other.currentTime) > 0.1) {{
        other.currentTime = v.currentTime;
        if (other.paused) other.play();
      }}
    }});
  }});
  v.addEventListener('ended', () => handleEnded(v));
}});
</script>
<hr>
<p><strong>Results:</strong> {result_text}</p>
"""
  return html


def main():
  parser = argparse.ArgumentParser(description='Compare two videos and generate HTML diff report')
  parser.add_argument('video1', help='First video file')
  parser.add_argument('video2', help='Second video file')
  parser.add_argument('output', nargs='?', default='diff.html', help='Output HTML file (default: diff.html)')
  parser.add_argument("--basedir", type=str, help="Base directory for output", default="")
  parser.add_argument('--no-open', action='store_true', help='Do not open HTML report in browser')

  args = parser.parse_args()

  os.makedirs(DIFF_OUT_DIR, exist_ok=True)

  print("=" * 60)
  print("VIDEO DIFF - HTML REPORT")
  print("=" * 60)
  print(f"Video 1: {args.video1}")
  print(f"Video 2: {args.video2}")
  print(f"Output: {args.output}")
  print()

  # Create diff video
  diff_video_path = os.path.join(os.path.dirname(args.output), DIFF_OUT_DIR / "diff.mp4")
  create_diff_video(args.video1, args.video2, diff_video_path)

  different_frames, frame_data, frame_counts = find_differences(args.video1, args.video2)

  if different_frames is None:
    sys.exit(1)

  print()
  print("Generating HTML report...")
  html = generate_html_report((args.video1, args.video2), args.basedir, different_frames, frame_data, frame_counts)

  with open(DIFF_OUT_DIR / args.output, 'w') as f:
    f.write(html)

  # Open in browser by default
  if not args.no_open:
    print(f"Opening {args.output} in browser...")
    webbrowser.open(f'file://{os.path.abspath(DIFF_OUT_DIR / args.output)}')

  extra_frames = abs(frame_counts[0] - frame_counts[1])
  return 0 if (len(different_frames) + extra_frames) == 0 else 1


if __name__ == "__main__":
  sys.exit(main())
