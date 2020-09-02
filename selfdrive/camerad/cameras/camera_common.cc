#include <assert.h>
#include <stdio.h>
#if defined(QCOM) && !defined(QCOM_REPLAY)
#include "cameras/camera_qcom.h"
#elif QCOM2
#include "cameras/camera_qcom2.h"
#elif WEBCAM
#include "cameras/camera_webcam.h"
#else
#include "cameras/camera_frame_stream.h"
#endif

#include <libyuv.h>

#include "clutil.h"
#include "common/swaglog.h"
#include "common/visionimg.h"
#include "imgproc/utils.h"

static cl_program build_debayer_program(cl_device_id device_id, cl_context context, const CameraInfo *ci, const CameraBuf *b) {
  char args[4096];
  snprintf(args, sizeof(args),
           "-cl-fast-relaxed-math -cl-denorms-are-zero "
           "-DFRAME_WIDTH=%d -DFRAME_HEIGHT=%d -DFRAME_STRIDE=%d "
           "-DRGB_WIDTH=%d -DRGB_HEIGHT=%d -DRGB_STRIDE=%d "
           "-DBAYER_FLIP=%d -DHDR=%d",
           ci->frame_width, ci->frame_height, ci->frame_stride,
           b->rgb_width, b->rgb_height, b->rgb_stride,
           ci->bayer_flip, ci->hdr);
#ifdef QCOM2
  return CLU_LOAD_FROM_FILE(context, device_id, "cameras/real_debayer.cl", args);
#else
  return CLU_LOAD_FROM_FILE(context, device_id, "cameras/debayer.cl", args);
#endif
}

void CameraBuf::init(cl_device_id device_id, cl_context context, CameraState *s, const char *name) {
  const CameraInfo *ci = &s->ci;
  camera_bufs = new VisionBuf[FRAME_BUF_COUNT];
  for (int i = 0; i < FRAME_BUF_COUNT; i++) {
    camera_bufs[i] = visionbuf_allocate_cl(s->frame_size, device_id, context);
  }

  rgb_width = ci->frame_width;
  rgb_height = ci->frame_height;
#ifndef QCOM2
  if (ci->bayer) {
    rgb_width = ci->frame_width / 2;
    rgb_height = ci->frame_height / 2;
  }
#endif

  if (ci->bayer) {
    // debayering does a 2x downscale
    yuv_transform = transform_scale_buffer(s->transform, 0.5);
  } else {
    yuv_transform = s->transform;
  }

  for (int i = 0; i < UI_BUF_COUNT; i++) {
    VisionImg img = visionimg_alloc_rgb24(rgb_width, rgb_height, &rgb_bufs[i]);
    visionbuf_to_cl(&rgb_bufs[i], device_id, context);
    if (i == 0) {
      rgb_stride = img.stride;
    }
  }
  tbuffer_init(&ui_tb, UI_BUF_COUNT, name);

  // yuv back for recording and orbd
  pool_init(&yuv_pool, YUV_COUNT);
  yuv_tb = pool_get_tbuffer(&yuv_pool);  //only for visionserver...

  yuv_width = rgb_width;
  yuv_height = rgb_height;
  yuv_buf_size = rgb_width * rgb_height * 3 / 2;

  for (int i = 0; i < YUV_COUNT; i++) {
    yuv_ion[i] = visionbuf_allocate_cl(yuv_buf_size, device_id, context);
    yuv_bufs[i].y = (uint8_t *)yuv_ion[i].addr;
    yuv_bufs[i].u = yuv_bufs[i].y + (yuv_width * yuv_height);
    yuv_bufs[i].v = yuv_bufs[i].u + (yuv_width / 2 * yuv_height / 2);
  }

  int err;
  if (ci->bayer) {
    cl_program prg_debayer = build_debayer_program(device_id, context, ci, this);
    krnl_debayer = clCreateKernel(prg_debayer, "debayer10", &err);
    assert(err == 0);
    assert(clReleaseProgram(prg_debayer) == 0);
  }

  rgb_to_yuv_init(&rgb_to_yuv_state, context, device_id, yuv_width, yuv_height, rgb_stride);

#ifdef __APPLE__
  q = clCreateCommandQueue(context, device_id, 0, &err);
#else
  const cl_queue_properties props[] = {0};  //CL_QUEUE_PRIORITY_KHR, CL_QUEUE_PRIORITY_HIGH_KHR, 0};
  q = clCreateCommandQueueWithProperties(context, device_id, props, &err);
#endif
  assert(err == 0);
}

void CameraBuf::free() {
  for (int i = 0; i < FRAME_BUF_COUNT; i++) {
    visionbuf_free(&camera_bufs[i]);
  }
  delete[] camera_bufs;

  for (int i = 0; i < UI_BUF_COUNT; i++) {
    visionbuf_free(&rgb_bufs[i]);
  }
  for (int i = 0; i < YUV_COUNT; i++) {
    visionbuf_free(&yuv_ion[i]);
  }
  clReleaseKernel(krnl_debayer);
  clReleaseCommandQueue(q);
}

bool CameraBuf::acquire(CameraState *s) {
  const int buf_idx = tbuffer_acquire(&s->camera_tb);
  if (buf_idx < 0) {
    return false;
  }
  const FrameMetadata &frame_data = s->camera_bufs_metadata[buf_idx];
  if (frame_data.frame_id == -1) {
    LOGE("no frame data? wtf");
    tbuffer_release(&s->camera_tb, buf_idx);
    return false;
  }

  cur_rgb_idx = tbuffer_select(&ui_tb);
  cur_rgb_buf = &rgb_bufs[cur_rgb_idx];
  
  cl_event debayer_event;
  cl_mem camrabuf_cl = camera_bufs[buf_idx].buf_cl;
  if (s->ci.bayer) {
    assert(clSetKernelArg(krnl_debayer, 0, sizeof(cl_mem), &camrabuf_cl) == 0);
    assert(clSetKernelArg(krnl_debayer, 1, sizeof(cl_mem), &cur_rgb_buf->buf_cl) == 0);
#ifdef QCOM2
    assert(clSetKernelArg(krnl_debayer, 2, s->debayer_cl_localMemSize, 0) == 0);
    assert(clEnqueueNDRangeKernel(q, krnl_debayer, 2, NULL,
                                  s->debayer_cl_globalWorkSize, s->debayer_cl_localWorkSize,
                                  0, 0, &debayer_event) == 0);
#else
    float digital_gain = s->digital_gain;
    if ((int)digital_gain == 0) {
      digital_gain = 1.0;
    }
    assert(clSetKernelArg(krnl_debayer, 2, sizeof(float), &digital_gain) == 0);
    const size_t debayer_work_size = rgb_height;  // doesn't divide evenly, is this okay?
    const size_t debayer_local_work_size = 128;
    assert(clEnqueueNDRangeKernel(q, krnl_debayer, 1, NULL,
                                  &debayer_work_size, &debayer_local_work_size, 0, 0, &debayer_event) == 0);
#endif
  } else {
    assert(cur_rgb_buf->len >= s->frame_size);
    assert(rgb_stride == s->ci.frame_stride);
    assert(clEnqueueCopyBuffer(q, camrabuf_cl, cur_rgb_buf->buf_cl, 0, 0,
                               cur_rgb_buf->len, 0, 0, &debayer_event) == 0);
  }

  clWaitForEvents(1, &debayer_event);
  clReleaseEvent(debayer_event);

  tbuffer_release(&s->camera_tb, buf_idx);
  visionbuf_sync(cur_rgb_buf, VISIONBUF_SYNC_FROM_DEVICE);

  cur_yuv_idx = pool_select(&yuv_pool);
  yuv_metas[cur_yuv_idx] = frame_data;
  rgb_to_yuv_queue(&rgb_to_yuv_state, q, cur_rgb_buf->buf_cl, yuv_ion[cur_yuv_idx].buf_cl);
  visionbuf_sync(&yuv_ion[cur_yuv_idx], VISIONBUF_SYNC_FROM_DEVICE);

  // keep another reference around till were done processing
  pool_acquire(&yuv_pool, cur_yuv_idx);
  pool_push(&yuv_pool, cur_yuv_idx);

  cur_yuv_buf = &yuv_bufs[cur_yuv_idx];
  cur_yuv_ion_buf = &yuv_ion[cur_yuv_idx];
  return true;
}

void CameraBuf::release() {
  tbuffer_dispatch(&ui_tb, cur_rgb_idx);
  pool_release(&yuv_pool, cur_yuv_idx);
}

void CameraBuf::stop() {
  tbuffer_stop(&ui_tb);
  pool_stop(&yuv_pool);
}

// common functions

void fill_frame_data(cereal::FrameData::Builder &framed, const FrameMetadata &frame_data, uint32_t cnt) {
  framed.setFrameId(frame_data.frame_id);
  framed.setEncodeId(cnt);
  framed.setTimestampEof(frame_data.timestamp_eof);
  framed.setFrameLength(frame_data.frame_length);
  framed.setIntegLines(frame_data.integ_lines);
  framed.setGlobalGain(frame_data.global_gain);
  framed.setLensPos(frame_data.lens_pos);
  framed.setLensSag(frame_data.lens_sag);
  framed.setLensErr(frame_data.lens_err);
  framed.setLensTruePos(frame_data.lens_true_pos);
  framed.setGainFrac(frame_data.gain_frac);
}

void common_camera_process_buf(MultiCameraState *s, const CameraBuf *b, int cnt, PubMaster* pm) {
  const FrameMetadata &frame_data = b->frameMetaData();
  if (pm != nullptr) {
    capnp::MallocMessageBuilder msg;
    cereal::Event::Builder event = msg.initRoot<cereal::Event>();
    event.setLogMonoTime(nanos_since_boot());
    auto framed = event.initFrame();
    fill_frame_data(framed, frame_data, cnt);

#if !defined(QCOM) && !defined(QCOM2)
    framed.setImage(kj::arrayPtr((const uint8_t*)b->cur_yuv_ion_buf->addr, b->yuv_buf_size));
#endif
    framed.setTransform(kj::ArrayPtr<const float>(&b->yuv_transform.v[0], 9));
    pm->send("frame", msg);
  }

#ifdef NOSCREEN
  if (frame_data.frame_id % 4 == 1) {
    sendrgb(s, (uint8_t *)b->cur_rgb_buf->addr, b->cur_rgb_buf->len, 0);
  }
#endif
}
