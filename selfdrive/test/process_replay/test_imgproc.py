import os
import numpy as np
import hashlib

import pyopencl as cl  # install with `PYOPENCL_CL_PRETEND_VERSION=2.0 pip install pyopencl`

from openpilot.system.hardware import PC, TICI
from openpilot.common.basedir import BASEDIR
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.system.camerad.snapshot.snapshot import yuv_to_rgb
from openpilot.tools.lib.logreader import LogReader

# TODO: check all sensors
TEST_ROUTE = "8345e3b82948d454|2022-05-04--13-45-33/0"

cam = DEVICE_CAMERAS[("tici", "ar0231")]
FRAME_WIDTH, FRAME_HEIGHT = (cam.dcam.width, cam.dcam.height)
FRAME_STRIDE = FRAME_WIDTH * 12 // 8 + 4

UV_WIDTH = FRAME_WIDTH // 2
UV_HEIGHT = FRAME_HEIGHT // 2
UV_SIZE = UV_WIDTH * UV_HEIGHT


def init_kernels(frame_offset=0):
  ctx = cl.create_some_context(interactive=False)

  with open(os.path.join(BASEDIR, 'system/camerad/cameras/process_raw.cl')) as f:
    build_args = f' -cl-fast-relaxed-math -cl-denorms-are-zero -cl-single-precision-constant -I{BASEDIR}/system/camerad/sensors ' + \
      f' -DFRAME_WIDTH={FRAME_WIDTH} -DFRAME_HEIGHT={FRAME_WIDTH} -DFRAME_STRIDE={FRAME_STRIDE} -DFRAME_OFFSET={frame_offset} ' + \
      f' -DRGB_WIDTH={FRAME_WIDTH} -DRGB_HEIGHT={FRAME_HEIGHT} -DYUV_STRIDE={FRAME_WIDTH} -DUV_OFFSET={FRAME_WIDTH*FRAME_HEIGHT}' + \
      ' -DSENSOR_ID=1 -DVIGNETTING=0 '
    if PC:
      build_args += ' -DHALF_AS_FLOAT=1 -cl-std=CL2.0'
    imgproc_prg = cl.Program(ctx, f.read()).build(options=build_args)

  return ctx, imgproc_prg

def proc_frame(ctx, imgproc_prg, data, rgb=False):
  q = cl.CommandQueue(ctx)

  yuv_buff = np.empty(FRAME_WIDTH * FRAME_HEIGHT + UV_SIZE * 2, dtype=np.uint8)

  cam_g = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=data)
  yuv_g = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, FRAME_WIDTH * FRAME_HEIGHT + UV_SIZE * 2)

  krn = imgproc_prg.process_raw
  krn.set_scalar_arg_dtypes([None, None, np.int32])
  local_worksize = (20, 20) if TICI else (4, 4)

  ev1 = krn(q, (FRAME_WIDTH//2, FRAME_HEIGHT//2), local_worksize, cam_g, yuv_g, 1)
  cl.enqueue_copy(q, yuv_buff, yuv_g, wait_for=[ev1]).wait()
  cl.enqueue_barrier(q)

  y = yuv_buff[:FRAME_WIDTH*FRAME_HEIGHT].reshape((FRAME_HEIGHT, FRAME_WIDTH))
  u = yuv_buff[FRAME_WIDTH*FRAME_HEIGHT::2].reshape((UV_HEIGHT, UV_WIDTH))
  v = yuv_buff[FRAME_WIDTH*FRAME_HEIGHT+1::2].reshape((UV_HEIGHT, UV_WIDTH))

  if rgb:
    return yuv_to_rgb(y, u, v)
  else:
    return y, u, v


def imgproc_replay(lr):
  ctx, imgproc_prg = init_kernels()

  frames = []
  for m in lr:
    if m.which() == 'roadCameraState':
      cs = m.roadCameraState
      if cs.image:
        data = np.frombuffer(cs.image, dtype=np.uint8)
        img = proc_frame(ctx, imgproc_prg, data)

        frames.append(img)

  return frames


if __name__ == "__main__":
  # load logs
  lr = list(LogReader(TEST_ROUTE))
  # run replay
  out_frames = imgproc_replay(lr)

  all_pix = np.concatenate([np.concatenate([d.flatten() for d in f]) for f in out_frames])
  pix_hash = hashlib.sha1(all_pix).hexdigest()

  with open('imgproc_replay_ref_hash') as f:
    ref_hash = f.read()

  if pix_hash != ref_hash:
    print("result changed! please check kernel")
    print(f"ref: {ref_hash}")
    print(f"new: {pix_hash}")
  else:
    print("test passed")
