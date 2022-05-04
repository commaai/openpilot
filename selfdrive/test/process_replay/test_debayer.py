#!/usr/bin/env python3
import os
import sys
import subprocess
import bz2
import numpy as np

from common.basedir import BASEDIR
from selfdrive.hardware import PC, TICI
from selfdrive.test.openpilotci import BASE_URL, get_url
from selfdrive.version import get_commit
from tools.lib.logreader import LogReader
from tools.lib.filereader import FileReader

try:
  import pyopencl as cl
except ImportError:
  print("failed to import pyopencl, installing...")
  env = os.environ.copy()
  env["PYOPENCL_CL_PRETEND_VERSION"] = "2.0"
  subprocess.run(["pip", "install", "pyopencl"], env=env, shell=True, check=True)
  import pyopencl as cl

TEST_ROUTE = "8345e3b82948d454|2022-05-04--13-45-33"
SEGMENT = 0

FRAME_WIDTH = 1928
FRAME_HEIGHT = 1208
FRAME_STRIDE = 2896

def get_frame_fn(ref_commit, test_route, tici=True):
  return f"{test_route}_debayer{'_tici' if tici else ''}_{ref_commit}.bz2"

def bzip_frames(frames):
  data = np.array(frames).tobytes()
  return bz2.compress(data)

def unbzip_frames(url):
  with FileReader(url) as f:
    dat = f.read()

  data = bz2.decompress(dat)
  res = np.frombuffer(data, dtype=np.uint8)

  return res.reshape((-1, FRAME_HEIGHT, FRAME_WIDTH, 3))

def debayer_frame(data):
  ctx = cl.create_some_context()
  q = cl.CommandQueue(ctx)

  with open(os.path.join(BASEDIR, 'selfdrive/camerad/cameras/real_debayer.cl')) as f:
    build_args = ' -cl-fast-relaxed-math -cl-denorms-are-zero -cl-single-precision-constant' + \
      f' -DFRAME_STRIDE={FRAME_STRIDE} -DRGB_WIDTH={FRAME_WIDTH} -DRGB_HEIGHT={FRAME_HEIGHT} -DCAM_NUM=0'
    if PC:
      build_args += ' -DHALF_AS_FLOAT=1'
    debayer_prg = cl.Program(ctx, f.read()).build(options=build_args)

  rgb_old_buff = np.empty(FRAME_WIDTH * FRAME_HEIGHT * 3, dtype=np.uint8)

  cam_g = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=data)
  rgb_wg = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, FRAME_WIDTH * FRAME_HEIGHT * 3)

  ev1 = debayer_prg.debayer10(q, (FRAME_WIDTH, FRAME_HEIGHT), (8, 8), cam_g, rgb_wg, cl.LocalMemory(400), np.float32(42))
  ev2 = cl.enqueue_copy(q, rgb_old_buff, rgb_wg, wait_for=[ev1])
  ev2.wait()

  cl.enqueue_barrier(q)

  res = rgb_old_buff.reshape((FRAME_HEIGHT, FRAME_WIDTH, 3))
  res[:, :, [2, 0]] = res[:, :, [0, 2]]

  return res

def debayer_replay(lr):
  rgb_frames = []

  for m in lr:
    if m.which() == 'roadCameraState':
      cs = m.roadCameraState
      if cs.image:
        data = np.frombuffer(cs.image, dtype=np.uint8)
        img = debayer_frame(data)

        rgb_frames.append(img)

  return np.array(rgb_frames)

if __name__ == "__main__":
  update = "--update" in sys.argv
  replay_dir = os.path.dirname(os.path.abspath(__file__))
  ref_commit_fn = os.path.join(replay_dir, "debayer_replay_ref_commit")

  # load logs
  lr = list(LogReader(get_url(TEST_ROUTE, SEGMENT)))

  # run replay
  frames = debayer_replay(lr)

  # get diff
  failed = False
  diff = ''
  diff_file = ''
  if not update:
    with open(ref_commit_fn) as f:
      ref_commit = f.read().strip()
    frame_fn = get_frame_fn(ref_commit, TEST_ROUTE, tici=TICI)

    try:
      cmp_frames = unbzip_frames(BASE_URL + frame_fn)

      if frames.shape != cmp_frames.shape:
        failed = True
        diff += 'frame shapes not equal\n'
        diff += f'{ref_commit}: {cmp_frames.shape}\n'
        diff += f'HEAD: {frames.shape}\n'
      elif not np.array_equal(frames, cmp_frames):
        failed = True
        if np.allclose(frames, cmp_frames, atol=1):
          diff += 'frames not equal, but are all close\n'
        else:
          diff += 'frames not equal\n'

        frame_diff = np.abs(np.subtract(frames, cmp_frames))
        diff_len = len(np.nonzero(frame_diff)[0])
        if diff_len > 1000000:
          diff += f'different at a large amount of pixels ({diff_len})\n'
        else:
          diff += 'different at (i, ref, HEAD):\n'
          for i in zip(*np.nonzero(frame_diff)):
            diff_file += f'{i}, {cmp_frames[i]}, {frames[i]}\n'

      if failed:
        print(diff)
        with open("debayer_diff.txt", "w") as f:
          diff_file = diff + diff_file
          f.write(diff_file)
    except Exception as e:
      print(str(e))
      failed = True

  # upload new refs
  if update or failed:
    from selfdrive.test.openpilotci import upload_file

    print("Uploading new refs")

    frames_bzip = bzip_frames(frames)

    new_commit = get_commit()
    frame_fn = os.path.join(replay_dir, get_frame_fn(new_commit, TEST_ROUTE, tici=TICI))
    with open(frame_fn, "wb") as f:  # type: ignore
      f.write(frames_bzip)

    try:
      upload_file(frame_fn, os.path.basename(frame_fn))
    except Exception as e:
      print("failed to upload", e)

  if update:
    with open(ref_commit_fn, 'w') as f:
      f.write(str(new_commit))

    print("\nNew ref commit: ", new_commit)

  sys.exit(int(failed))
