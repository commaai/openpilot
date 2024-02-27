#!/usr/bin/env python3
import os
import sys
import bz2
import numpy as np

import pyopencl as cl  # install with `PYOPENCL_CL_PRETEND_VERSION=2.0 pip install pyopencl`

from openpilot.system.hardware import PC, TICI
from openpilot.common.basedir import BASEDIR
from openpilot.tools.lib.openpilotci import BASE_URL
from openpilot.system.version import get_commit
from openpilot.system.camerad.snapshot.snapshot import yuv_to_rgb
from openpilot.tools.lib.logreader import LogReader
from openpilot.tools.lib.filereader import FileReader

TEST_ROUTE = "8345e3b82948d454|2022-05-04--13-45-33/0"

FRAME_WIDTH = 1928
FRAME_HEIGHT = 1208
FRAME_STRIDE = 2896

UV_WIDTH = FRAME_WIDTH // 2
UV_HEIGHT = FRAME_HEIGHT // 2
UV_SIZE = UV_WIDTH * UV_HEIGHT


def get_frame_fn(ref_commit, test_route, tici=True):
  return f"{test_route}_debayer{'_tici' if tici else ''}_{ref_commit}.bz2"


def bzip_frames(frames):
  data = b''
  for y, u, v in frames:
    data += y.tobytes()
    data += u.tobytes()
    data += v.tobytes()
  return bz2.compress(data)


def unbzip_frames(url):
  with FileReader(url) as f:
    dat = f.read()

  data = bz2.decompress(dat)

  res = []
  for y_start in range(0, len(data), FRAME_WIDTH * FRAME_HEIGHT + UV_SIZE * 2):
    u_start = y_start + FRAME_WIDTH * FRAME_HEIGHT
    v_start = u_start + UV_SIZE

    y = np.frombuffer(data[y_start: u_start], dtype=np.uint8).reshape((FRAME_HEIGHT, FRAME_WIDTH))
    u = np.frombuffer(data[u_start: v_start], dtype=np.uint8).reshape((UV_HEIGHT, UV_WIDTH))
    v = np.frombuffer(data[v_start: v_start + UV_SIZE], dtype=np.uint8).reshape((UV_HEIGHT, UV_WIDTH))

    res.append((y, u, v))

  return res


def init_kernels(frame_offset=0):
  ctx = cl.create_some_context(interactive=False)

  with open(os.path.join(BASEDIR, 'system/camerad/cameras/real_debayer.cl')) as f:
    build_args = ' -cl-fast-relaxed-math -cl-denorms-are-zero -cl-single-precision-constant' + \
      f' -DFRAME_STRIDE={FRAME_STRIDE} -DRGB_WIDTH={FRAME_WIDTH} -DRGB_HEIGHT={FRAME_HEIGHT} -DFRAME_OFFSET={frame_offset} -DCAM_NUM=0'
    if PC:
      build_args += ' -DHALF_AS_FLOAT=1 -cl-std=CL2.0'
    debayer_prg = cl.Program(ctx, f.read()).build(options=build_args)

  return ctx, debayer_prg

def debayer_frame(ctx, debayer_prg, data, rgb=False):
  q = cl.CommandQueue(ctx)

  yuv_buff = np.empty(FRAME_WIDTH * FRAME_HEIGHT + UV_SIZE * 2, dtype=np.uint8)

  cam_g = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=data)
  yuv_g = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, FRAME_WIDTH * FRAME_HEIGHT + UV_SIZE * 2)

  local_worksize = (20, 20) if TICI else (4, 4)
  ev1 = debayer_prg.debayer10(q, (UV_WIDTH, UV_HEIGHT), local_worksize, cam_g, yuv_g)
  cl.enqueue_copy(q, yuv_buff, yuv_g, wait_for=[ev1]).wait()
  cl.enqueue_barrier(q)

  y = yuv_buff[:FRAME_WIDTH*FRAME_HEIGHT].reshape((FRAME_HEIGHT, FRAME_WIDTH))
  u = yuv_buff[FRAME_WIDTH*FRAME_HEIGHT:FRAME_WIDTH*FRAME_HEIGHT+UV_SIZE].reshape((UV_HEIGHT, UV_WIDTH))
  v = yuv_buff[FRAME_WIDTH*FRAME_HEIGHT+UV_SIZE:].reshape((UV_HEIGHT, UV_WIDTH))

  if rgb:
    return yuv_to_rgb(y, u, v)
  else:
    return y, u, v


def debayer_replay(lr):
  ctx, debayer_prg = init_kernels()

  frames = []
  for m in lr:
    if m.which() == 'roadCameraState':
      cs = m.roadCameraState
      if cs.image:
        data = np.frombuffer(cs.image, dtype=np.uint8)
        img = debayer_frame(ctx, debayer_prg, data)

        frames.append(img)

  return frames


if __name__ == "__main__":
  update = "--update" in sys.argv
  replay_dir = os.path.dirname(os.path.abspath(__file__))
  ref_commit_fn = os.path.join(replay_dir, "debayer_replay_ref_commit")

  # load logs
  lr = list(LogReader(TEST_ROUTE))

  # run replay
  frames = debayer_replay(lr)

  # get diff
  failed = False
  diff = ''
  yuv_i = ['y', 'u', 'v']
  if not update:
    with open(ref_commit_fn) as f:
      ref_commit = f.read().strip()
    frame_fn = get_frame_fn(ref_commit, TEST_ROUTE, tici=TICI)

    try:
      cmp_frames = unbzip_frames(BASE_URL + frame_fn)

      if len(frames) != len(cmp_frames):
        failed = True
        diff += 'amount of frames not equal\n'

      for i, (frame, cmp_frame) in enumerate(zip(frames, cmp_frames, strict=True)):
        for j in range(3):
          fr = frame[j]
          cmp_f = cmp_frame[j]
          if fr.shape != cmp_f.shape:
            failed = True
            diff += f'frame shapes not equal for ({i}, {yuv_i[j]})\n'
            diff += f'{ref_commit}: {cmp_f.shape}\n'
            diff += f'HEAD: {fr.shape}\n'
          elif not np.array_equal(fr, cmp_f):
            failed = True
            if np.allclose(fr, cmp_f, atol=1):
              diff += f'frames not equal for ({i}, {yuv_i[j]}), but are all close\n'
            else:
              diff += f'frames not equal for ({i}, {yuv_i[j]})\n'

            frame_diff = np.abs(np.subtract(fr, cmp_f))
            diff_len = len(np.nonzero(frame_diff)[0])
            if diff_len > 10000:
              diff += f'different at a large amount of pixels ({diff_len})\n'
            else:
              diff += 'different at (frame, yuv, pixel, ref, HEAD):\n'
              for k in zip(*np.nonzero(frame_diff), strict=True):
                diff += f'{i}, {yuv_i[j]}, {k}, {cmp_f[k]}, {fr[k]}\n'

      if failed:
        print(diff)
        with open("debayer_diff.txt", "w") as f:
          f.write(diff)
    except Exception as e:
      print(str(e))
      failed = True

  # upload new refs
  if update or (failed and TICI):
    from openpilot.tools.lib.openpilotci import upload_file

    print("Uploading new refs")

    frames_bzip = bzip_frames(frames)

    new_commit = get_commit()
    frame_fn = os.path.join(replay_dir, get_frame_fn(new_commit, TEST_ROUTE, tici=TICI))
    with open(frame_fn, "wb") as f2:
      f2.write(frames_bzip)

    try:
      upload_file(frame_fn, os.path.basename(frame_fn))
    except Exception as e:
      print("failed to upload", e)

  if update:
    with open(ref_commit_fn, 'w') as f:
      f.write(str(new_commit))

    print("\nNew ref commit: ", new_commit)

  sys.exit(int(failed))
