#!/usr/bin/env python3
import difflib
import json
import os
import sys
import subprocess
import webbrowser
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass
from typing import Literal
from openpilot.common.basedir import BASEDIR

DIFF_OUT_DIR = Path(BASEDIR) / "selfdrive" / "ui" / "tests" / "diff" / "report"
HTML_TEMPLATE_PATH = Path(__file__).with_name("diff_template.html")

# extra frames of context to include before/after each diff chunk
# NOTE: adding this can make it difficult to view the exact frames that changed if they are small,
# but it could be useful for added context or to reduce any video flickering.
CHUNK_PADDING_BEFORE = 0
CHUNK_PADDING_AFTER = 0


def extract_framehashes(video_path: Path) -> list[str]:
  cmd = ['ffmpeg', '-nostdin', '-i', str(video_path), '-map', '0:v:0', '-vsync', '0', '-f', 'framehash', '-hash', 'md5', '-']
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


@dataclass
class DiffChunk:
  """Represents a contiguous chunk of differences between the two videos. Ranges (start-end) are inclusive."""
  type: Literal['replace', 'insert', 'delete']
  v1_start: int
  v1_end: int
  v1_count: int
  v2_start: int
  v2_end: int
  v2_count: int


def compute_diff_chunks(hashes1: list[str], hashes2: list[str]) -> list[DiffChunk]:
  """Use difflib to compute diff chunks from the two hash lists. Returns a list of DiffChunk objects."""
  matcher = difflib.SequenceMatcher(a=hashes1, b=hashes2, autojunk=False)
  diff_ops: list[list] = [list(op) for op in matcher.get_opcodes() if op[0] != 'equal']  # filter out equal chunks
  chunks: list[DiffChunk] = []
  for tag, i1, i2, j1, j2 in diff_ops:
    chunks.append(DiffChunk(
      type=tag,
      v1_start=i1, v1_end=i2 - 1, v1_count=i2 - i1,
      v2_start=j1, v2_end=j2 - 1, v2_count=j2 - j1,
    ))
  return chunks


def create_diff_video(video1: Path, video2: Path, output: Path) -> None:
  """Create a diff video of two clips using ffmpeg blend filter with difference mode."""
  cmd = ['ffmpeg', '-nostdin', '-i', str(video1), '-i', str(video2), '-filter_complex', 'blend=all_mode=difference', '-vsync', '0', '-y', str(output)]
  subprocess.run(cmd, capture_output=True, check=True)


def get_video_fps(video_path: Path) -> float:
  """Return fps for a video file."""
  cmd = ['ffprobe', '-select_streams', 'v:0', '-show_entries', 'stream=r_frame_rate', '-of', 'json', str(video_path)]
  result = subprocess.run(cmd, capture_output=True, text=True, check=True)
  info = json.loads(result.stdout)['streams'][0]
  num, den = info['r_frame_rate'].split('/')
  return int(num) / int(den)


def extract_clip(video_path: Path, start_frame: int, end_frame: int, output_path: Path, fps: float) -> int:
  """Extract [start_frame, end_frame] plus padding before/after into *output_path*. Returns the actual number of frames written."""
  padded_start = max(0, start_frame - CHUNK_PADDING_BEFORE)
  padding_before = start_frame - padded_start
  total_frames = (end_frame - start_frame + 1) + padding_before + CHUNK_PADDING_AFTER
  start_time = padded_start / fps
  cmd = ['ffmpeg', '-nostdin', '-i', str(video_path), '-ss', f"{start_time:.6f}", '-frames:v', str(total_frames), '-vsync', '0', '-y', str(output_path)]
  subprocess.run(cmd, capture_output=True, check=True)
  return total_frames


def generate_thumbnail(video_path: Path, frame: int, out_path: Path, fps: float) -> None:
  """Create a single-frame thumbnail at the given frame index. File format is determined by the output extension (e.g. .jpg or .png)."""
  t = frame / fps
  cmd = ['ffmpeg', '-nostdin', '-i', str(video_path), '-ss', f"{t:.6f}", '-frames:v', '1', '-vsync', '0', '-y', str(out_path)]
  subprocess.run(cmd, capture_output=True, check=True)


def extract_chunk_clips(video1: Path, video2: Path, chunks: list[DiffChunk], fps: float, basedir: str, folder_name: str) -> list[dict]:
  """For each diff chunk, extract clips from video1, video2, a diff video (if both are available), and a thumbnail image."""
  output_dir = DIFF_OUT_DIR / folder_name
  os.makedirs(output_dir, exist_ok=True)

  def get_rel_path(p: Path) -> str:
    """ Return path relative to the basedir."""
    return os.path.join(basedir, folder_name, p.name)

  def process_chunk(i: int, chunk: DiffChunk) -> dict:
    chunk_type = chunk.type
    v1_start, v1_end, v1_count = chunk.v1_start, chunk.v1_end, chunk.v1_count
    v2_start, v2_end, v2_count = chunk.v2_start, chunk.v2_end, chunk.v2_count
    clips: dict[str, str | None] = {'video1': None, 'video2': None, 'diff': None}

    v1_clip = output_dir / f"{i:03d}_video1.mp4"
    v2_clip = output_dir / f"{i:03d}_video2.mp4"

    # Parallelize video1/video2 clip extractions within each chunk
    with ThreadPoolExecutor(max_workers=2) as executor:
      futures = []
      if chunk_type != 'insert':
        # video 1 clip
        futures.append(executor.submit(extract_clip, video1, v1_start, v1_end, v1_clip, fps))
        clips['video1'] = get_rel_path(v1_clip)
      if chunk_type != 'delete':
        # video 2 clip
        futures.append(executor.submit(extract_clip, video2, v2_start, v2_end, v2_clip, fps))
        clips['video2'] = get_rel_path(v2_clip)
      for future in futures:
        future.result()

    # diff clip
    diff_clip = output_dir / f"{i:03d}_diff.mp4"
    if chunk_type == 'replace':
      create_diff_video(v1_clip, v2_clip, diff_clip)
      clips['diff'] = get_rel_path(diff_clip)

    # thumbnail (middle frame of the diff content inside the clip)
    padding_used = min((v1_start if chunk_type != 'insert' else v2_start), CHUNK_PADDING_BEFORE)
    content_count = v1_count if chunk_type != 'insert' else v2_count
    thumb_frame = padding_used + content_count // 2
    thumb_ext = 'png' if chunk_type == 'replace' else 'jpg'  # Use PNG for the diff thumbnails for clarity; JPG is smaller for the other thumbnails
    thumb_path = output_dir / f"{i:03d}_thumb.{thumb_ext}"
    thumb_source = diff_clip if chunk_type == 'replace' else (v1_clip if chunk_type == 'delete' else v2_clip)
    generate_thumbnail(thumb_source, thumb_frame, thumb_path, fps)

    return {
      'index': i, 'type': chunk_type, 'clips': clips, 'thumb': get_rel_path(thumb_path),
      'v1_start': v1_start, 'v1_end': v1_end, 'v1_count': v1_count,
      'v2_start': v2_start, 'v2_end': v2_end, 'v2_count': v2_count,
    }

  # Process chunks in parallel with a thread pool
  max_workers = min(8, len(chunks))
  print(f"  Processing {len(chunks)} chunks with {max_workers} threads...")
  results = []
  with ThreadPoolExecutor(max_workers) as executor:
    futures = [executor.submit(process_chunk, i, chunk) for i, chunk in enumerate(chunks)]
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing chunks", unit="chunk", disable=bool(os.getenv("CI"))):
      results.append(future.result())
  # results will be out of order due to parallel processing, so sort them back to the original order
  clip_sets = sorted(results, key=lambda x: x['index'])

  return clip_sets


def generate_html_report(
  videos: tuple[Path, Path], basedir: str, diff_frame_count: int, frame_counts: tuple[int, int], diff_video_name: str, clip_sets: list[dict]
) -> str:
  total_frames = max(frame_counts)
  frame_delta = frame_counts[1] - frame_counts[0]

  result_text = (
    f"✅ Videos are identical! ({total_frames} frames)"
    if diff_frame_count == 0
    else f"❌ Found {diff_frame_count} different frames out of {total_frames} total ({diff_frame_count / total_frames * 100:.1f}%)."
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
    "CHUNKS_JSON": json.dumps(clip_sets),
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

  video1 = Path(args.video1)
  video2 = Path(args.video2)
  missing = [str(p) for p in (video1, video2) if not p.exists()]
  if missing:
    parser.error(f"Video file(s) not found: {', '.join(missing)}")

  output_stem = Path(args.output).stem
  diff_video_name = f"{output_stem}.mp4"
  chunks_folder_name = f"{output_stem}-chunks"

  os.makedirs(DIFF_OUT_DIR, exist_ok=True)

  print("=" * 60)
  print("UI VIDEO DIFF REPORT")
  print("=" * 60)
  print(f"Video 1:      {video1}")
  print(f"Video 2:      {video2}")
  print(f"HTML output:  {args.output}")
  print(f"Diff video:   {diff_video_name}")
  print(f"Chunks dir:   {chunks_folder_name}")
  print()

  print("[1/5] Starting full video diff generation in background thread...")
  diff_thread = threading.Thread(target=create_diff_video, args=(video1, video2, DIFF_OUT_DIR / diff_video_name))
  diff_thread.start()

  print("[2/5] Hashing frames...")
  hashes1, hashes2 = get_video_frame_hashes(video1, video2)
  frame_counts = (len(hashes1), len(hashes2))
  print(f"  Found {frame_counts[0]} frames in video 1 and {frame_counts[1]} frames in video 2.")

  print("[3/5] Computing diff chunks...")
  chunks = compute_diff_chunks(hashes1, hashes2)
  diff_frame_count = sum(max(c.v1_count, c.v2_count) for c in chunks)  # TODO: We could be more precise by reporting changes, inserts, and deletions separately.
  print(f"  Found {len(chunks)} diff chunk(s) with a total of {diff_frame_count} different frames.")

  clip_sets = []
  if chunks:
    print(f"[4/5] Extracting {len(chunks)} diff chunk(s)...")
    print("  Getting video fps...", end=' ')
    fps1, fps2 = get_video_fps(video1), get_video_fps(video2)
    if abs(fps1 - fps2) > 0.01:
      raise ValueError(f"Videos have different fps values: {fps1:.2f} vs {fps2:.2f}")
    print(f"{fps1:.2f} fps")
    clip_sets = extract_chunk_clips(video1, video2, chunks, fps1, args.basedir, chunks_folder_name)
  else:
    print("[4/5] No diff chunks found, skipping clip extraction.")

  print("[5/5] Generating HTML report...")
  html = generate_html_report((video1, video2), args.basedir, diff_frame_count, frame_counts, diff_video_name, clip_sets)

  output_path = DIFF_OUT_DIR / args.output
  with open(output_path, 'w') as f:
    f.write(html)

  print(f"  Report generated at: {output_path}")

  # Open in browser by default
  if not args.no_open:
    print(f"Opening {args.output} in browser...")
    webbrowser.open(f'file://{os.path.abspath(output_path)}')

  if diff_thread.is_alive():
    print("Waiting for diff video generation to finish...")
  diff_thread.join()

  return 0 if diff_frame_count == 0 else 1


if __name__ == "__main__":
  sys.exit(main())
