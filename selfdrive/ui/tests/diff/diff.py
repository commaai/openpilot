#!/usr/bin/env python3
import os
import sys
import subprocess
import tempfile
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


def create_diff_video(video1, video2, output_path):
  """Create a diff video using ffmpeg blend filter with difference mode."""
  print("Creating diff video...")
  cmd = ['ffmpeg', '-i', video1, '-i', video2, '-filter_complex', '[0:v]blend=all_mode=difference', '-vsync', '0', '-y', output_path]
  subprocess.run(cmd, capture_output=True, check=True)


def find_differences(video1, video2):
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

    if len(frames1) != len(frames2):
      print(f"WARNING: Frame count mismatch: {len(frames1)} vs {len(frames2)}")
      min_frames = min(len(frames1), len(frames2))
      frames1 = frames1[:min_frames]
      frames2 = frames2[:min_frames]

    print(f"Comparing {len(frames1)} frames...")
    different_frames = []

    for i, (f1, f2) in enumerate(zip(frames1, frames2, strict=False)):
      is_different = not compare_frames(f1, f2)
      if is_different:
        different_frames.append(i)

    return different_frames, len(frames1)


def generate_html_report(video1, video2, basedir, different_frames, total_frames):
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

  result_text = (
    f"✅ Videos are identical! ({total_frames} frames)"
    if len(different_frames) == 0
    else f"❌ Found {len(different_frames)} different frames out of {total_frames} total ({(len(different_frames) / total_frames * 100):.1f}%)"
  )

  html = f"""<h2>UI Diff</h2>
<table>
<tr>
<td width='33%'>
  <p><strong>Video 1</strong></p>
  <video id='video1' width='100%' autoplay muted loop onplay='syncVideos()'>
    <source src='{os.path.join(basedir, os.path.basename(video1))}' type='video/mp4'>
    Your browser does not support the video tag.
  </video>
</td>
<td width='33%'>
  <p><strong>Video 2</strong></p>
  <video id='video2' width='100%' autoplay muted loop onplay='syncVideos()'>
    <source src='{os.path.join(basedir, os.path.basename(video2))}' type='video/mp4'>
    Your browser does not support the video tag.
  </video>
</td>
<td width='33%'>
  <p><strong>Pixel Diff</strong></p>
  <video id='diffVideo' width='100%' autoplay muted loop>
    <source src='{os.path.join(basedir, 'diff.mp4')}' type='video/mp4'>
    Your browser does not support the video tag.
  </video>
</td>
</tr>
</table>
<script>
function syncVideos() {{
  const video1 = document.getElementById('video1');
  const video2 = document.getElementById('video2');
  const diffVideo = document.getElementById('diffVideo');
  video1.currentTime = video2.currentTime = diffVideo.currentTime;
}}
video1.addEventListener('timeupdate', () => {{
  if (Math.abs(video1.currentTime - video2.currentTime) > 0.1) {{
    video2.currentTime = video1.currentTime;
  }}
  if (Math.abs(video1.currentTime - diffVideo.currentTime) > 0.1) {{
    diffVideo.currentTime = video1.currentTime;
  }}
}});
video2.addEventListener('timeupdate', () => {{
  if (Math.abs(video2.currentTime - video1.currentTime) > 0.1) {{
    video1.currentTime = video2.currentTime;
  }}
  if (Math.abs(video2.currentTime - diffVideo.currentTime) > 0.1) {{
    diffVideo.currentTime = video2.currentTime;
  }}
}});
diffVideo.addEventListener('timeupdate', () => {{
  if (Math.abs(diffVideo.currentTime - video1.currentTime) > 0.1) {{
    video1.currentTime = diffVideo.currentTime;
    video2.currentTime = diffVideo.currentTime;
  }}
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

  different_frames, total_frames = find_differences(args.video1, args.video2)

  if different_frames is None:
    sys.exit(1)

  print()
  print("Generating HTML report...")
  html = generate_html_report(args.video1, args.video2, args.basedir, different_frames, total_frames)

  with open(DIFF_OUT_DIR / args.output, 'w') as f:
    f.write(html)

  # Open in browser by default
  if not args.no_open:
    print(f"Opening {args.output} in browser...")
    webbrowser.open(f'file://{os.path.abspath(DIFF_OUT_DIR / args.output)}')

  return 0 if len(different_frames) == 0 else 1


if __name__ == "__main__":
  sys.exit(main())
