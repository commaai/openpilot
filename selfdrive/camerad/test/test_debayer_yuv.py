#!/usr/bin/env python

import os
os.environ['PYOPENCL_NO_CACHE'] = '1'

import numpy as np
import pyopencl as cl  # install with `PYOPENCL_CL_PRETEND_VERSION="2.0" pip install pyopencl`

np.set_printoptions(edgeitems=10, linewidth=180)

w = 16  # TODO: larger
h = 16
stride = 20

ctx = cl.create_some_context()
q = cl.CommandQueue(ctx)

with open('selfdrive/camerad/cameras/real_debayer.cl') as f:
  build_args = ' -cl-fast-relaxed-math -cl-denorms-are-zero -cl-single-precision-constant' + \
    f' -DFRAME_STRIDE={stride} -DRGB_WIDTH={w} -DRGB_HEIGHT={h} -DCAM_NUM=0'
  real_debayer_prg = cl.Program(ctx, f.read()).build(options=build_args)

with open('selfdrive/camerad/test/rgb_debayer.cl') as f:
  build_args = ' -cl-fast-relaxed-math -cl-denorms-are-zero -cl-single-precision-constant' + \
    f' -DFRAME_STRIDE={stride} -DRGB_WIDTH={w} -DRGB_HEIGHT={h} -DCAM_NUM=0'
  rgb_debayer_prg = cl.Program(ctx, f.read()).build(options=build_args)

with open('selfdrive/camerad/transforms/rgb_to_yuv.cl') as f:
  build_args = f' -cl-fast-relaxed-math -cl-denorms-are-zero -DWIDTH={w} -DHEIGHT={h}' + \
    f' -DUV_WIDTH={w//2} -DUV_HEIGHT={h//2} -DRGB_STRIDE={w*3} -DRGB_SIZE={w*h}'
  rgb_to_yuv_prg = cl.Program(ctx, f.read()).build(options=build_args)

cam_frame = np.random.randint(0, 256, (stride, h), dtype=np.uint8)

cam_buff = cam_frame.flatten()
cam_buff2 = cam_buff.copy()
rgb_old_buff = np.empty(w * h * 3, dtype=np.uint8)
yuv_old_buff = np.empty(w * h + (w // 2) * (h // 2) * 2, dtype=np.uint8)
yuv_new_buff = np.empty(w * h + (w // 2) * (h // 2) * 2, dtype=np.uint8)

cam_g = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=cam_buff)
cam2_g = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=cam_buff2)
rgb_wg = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, w * h * 3)
yuv_old_g = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, yuv_old_buff.nbytes)
yuv_new_g = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, yuv_new_buff.nbytes)

ev1 = rgb_debayer_prg.debayer10(q, (w, h), None, cam_g, rgb_wg, cl.LocalMemory(640))
cl.enqueue_copy(q, rgb_old_buff, rgb_wg, wait_for=[ev1]).wait()

cl.enqueue_barrier(q)

rgb_rg = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=rgb_old_buff)

cl.enqueue_barrier(q)

ev2 = rgb_to_yuv_prg.rgb_to_yuv(q, (w // 4, h // 4), None, rgb_rg, yuv_old_g, wait_for=[ev1])
ev3 = real_debayer_prg.debayer10(q, (w // 2, h // 2), (4, 4), cam2_g, yuv_new_g, cl.LocalMemory(2000))

cl.enqueue_barrier(q)

cl.enqueue_copy(q, yuv_old_buff, yuv_old_g, wait_for=[ev2]).wait()
cl.enqueue_copy(q, yuv_new_buff, yuv_new_g, wait_for=[ev3]).wait()

cl.enqueue_barrier(q)

y_old = yuv_old_buff[:w*h].reshape((h, w))
u_old = yuv_old_buff[w*h:w*h+(w//2)*(h//2)].reshape((h//2, w//2))
v_old = yuv_old_buff[w*h+(w//2)*(h//2):].reshape((h//2, w//2))

y_new = yuv_new_buff[:w*h].reshape((h, w))
u_new = yuv_new_buff[w*h:w*h+(w//2)*(h//2)].reshape((h//2, w//2))
v_new = yuv_new_buff[w*h+(w//2)*(h//2):].reshape((h//2, w//2))

assert np.allclose(y_old[2:-2, 2:-2], y_new[2:-2, 2:-2])
assert np.allclose(u_old[2:-2, 2:-2], u_new[2:-2, 2:-2])
assert np.allclose(v_old[2:-2, 2:-2], v_new[2:-2, 2:-2])
