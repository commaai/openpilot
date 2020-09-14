#include <thread>
#include <stdio.h>
#include <signal.h>

#if defined(QCOM) && !defined(QCOM_REPLAY)
#include "cameras/camera_qcom.h"
#elif QCOM2
#include "cameras/camera_qcom2.h"
#elif WEBCAM
#include "cameras/camera_webcam.h"
#else
#include "cameras/camera_frame_stream.h"
#endif

#include <czmq.h>
#include <libyuv.h>

#include "clutil.h"
#include "common/ipc.h"
#include "common/params.h"
#include "common/swaglog.h"
#include "common/util.h"
#include "common/visionipc.h"
#include "imgproc/utils.h"

#define MAX_CLIENTS 6

extern "C" {
volatile sig_atomic_t do_exit = 0;
}

void set_do_exit(int sig) {
  do_exit = 1;
}

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

void camerabuf_init(CameraBuf *b, cl_device_id device_id, cl_context context, CameraState *s, const char *name) {
  const CameraInfo *ci = &s->ci;
  b->camera_bufs = new VisionBuf[FRAME_BUF_COUNT];
  for (int i = 0; i < FRAME_BUF_COUNT; i++) {
    b->camera_bufs[i] = visionbuf_allocate_cl(s->frame_size, device_id, context);
  }

  b->rgb_width = ci->frame_width;
  b->rgb_height = ci->frame_height;
#ifndef QCOM2
  if (ci->bayer) {
    b->rgb_width = ci->frame_width / 2;
    b->rgb_height = ci->frame_height / 2;
  }
#endif

  if (ci->bayer) {
    // debayering does a 2x downscale
    b->yuv_transform = transform_scale_buffer(s->transform, 0.5);
  } else {
    b->yuv_transform = s->transform;
  }

  for (int i = 0; i < UI_BUF_COUNT; i++) {
    VisionImg img = visionimg_alloc_rgb24(b->rgb_width, b->rgb_height, &b->rgb_bufs[i]);
    visionbuf_to_cl(&b->rgb_bufs[i], device_id, context);
    if (i == 0) {
      b->rgb_stride = img.stride;
    }
  }
  tbuffer_init(&b->ui_tb, UI_BUF_COUNT, name);

  // yuv back for recording and orbd
  pool_init(&b->yuv_pool, YUV_COUNT);
  b->yuv_tb = pool_get_tbuffer(&b->yuv_pool);  //only for visionserver...

  b->yuv_width = b->rgb_width;
  b->yuv_height = b->rgb_height;
  b->yuv_buf_size = b->rgb_width * b->rgb_height * 3 / 2;

  for (int i = 0; i < YUV_COUNT; i++) {
    b->yuv_ion[i] = visionbuf_allocate_cl(b->yuv_buf_size, device_id, context);
    b->yuv_bufs[i].y = (uint8_t *)b->yuv_ion[i].addr;
    b->yuv_bufs[i].u = b->yuv_bufs[i].y + (b->yuv_width * b->yuv_height);
    b->yuv_bufs[i].v = b->yuv_bufs[i].u + (b->yuv_width / 2 * b->yuv_height / 2);
  }

  int err;
  if (ci->bayer) {
    cl_program prg_debayer = build_debayer_program(device_id, context, ci, b);
    b->krnl_debayer = clCreateKernel(prg_debayer, "debayer10", &err);
    assert(err == 0);
    assert(clReleaseProgram(prg_debayer) == 0);
  }

  rgb_to_yuv_init(&b->rgb_to_yuv_state, context, device_id, b->yuv_width, b->yuv_height, b->rgb_stride);

#ifdef __APPLE__
  b->q = clCreateCommandQueue(context, device_id, 0, &err);
#else
  const cl_queue_properties props[] = {0};  //CL_QUEUE_PRIORITY_KHR, CL_QUEUE_PRIORITY_HIGH_KHR, 0};
  b->q = clCreateCommandQueueWithProperties(context, device_id, props, &err);
#endif
  assert(err == 0);
}

void camerabuf_free(CameraBuf *b) {
  for (int i = 0; i < FRAME_BUF_COUNT; i++) {
    visionbuf_free(&b->camera_bufs[i]);
  }
  delete[] b->camera_bufs;

  for (int i = 0; i < UI_BUF_COUNT; i++) {
    visionbuf_free(&b->rgb_bufs[i]);
  }
  for (int i = 0; i < YUV_COUNT; i++) {
    visionbuf_free(&b->yuv_ion[i]);
  }
  clReleaseKernel(b->krnl_debayer);
  clReleaseCommandQueue(b->q);
}

bool camerabuf_acquire(CameraBuf *b, CameraState *s) {
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

  b->cur_rgb_idx = tbuffer_select(&b->ui_tb);
  VisionBuf *rgb_buf = &b->rgb_bufs[b->cur_rgb_idx];

  cl_event debayer_event;
  cl_mem camrabuf_cl = b->camera_bufs[buf_idx].buf_cl;
  if (s->ci.bayer) {
    assert(clSetKernelArg(b->krnl_debayer, 0, sizeof(cl_mem), &camrabuf_cl) == 0);
    assert(clSetKernelArg(b->krnl_debayer, 1, sizeof(cl_mem), &rgb_buf->buf_cl) == 0);
#ifdef QCOM2
    assert(clSetKernelArg(b->krnl_debayer, 2, s->debayer_cl_localMemSize, 0) == 0);
    assert(clEnqueueNDRangeKernel(b->q, b->krnl_debayer, 2, NULL,
                                  s->debayer_cl_globalWorkSize, s->debayer_cl_localWorkSize,
                                  0, 0, &debayer_event) == 0);
#else
    float digital_gain = s->digital_gain;
    if ((int)digital_gain == 0) {
      digital_gain = 1.0;
    }
    assert(clSetKernelArg(b->krnl_debayer, 2, sizeof(float), &digital_gain) == 0);
    const size_t debayer_work_size = b->rgb_height;  // doesn't divide evenly, is this okay?
    const size_t debayer_local_work_size = 128;
    assert(clEnqueueNDRangeKernel(b->q, b->krnl_debayer, 1, NULL,
                                  &debayer_work_size, &debayer_local_work_size, 0, 0, &debayer_event) == 0);
#endif
  } else {
    assert(rgb_buf->len >= s->frame_size);
    assert(b->rgb_stride == s->ci.frame_stride);
    assert(clEnqueueCopyBuffer(b->q, camrabuf_cl, rgb_buf->buf_cl, 0, 0,
                               rgb_buf->len, 0, 0, &debayer_event) == 0);
  }

  clWaitForEvents(1, &debayer_event);
  clReleaseEvent(debayer_event);

  tbuffer_release(&s->camera_tb, buf_idx);
  visionbuf_sync(rgb_buf, VISIONBUF_SYNC_FROM_DEVICE);

  b->cur_yuv_idx = pool_select(&b->yuv_pool);
  b->yuv_metas[b->cur_yuv_idx] = frame_data;
  rgb_to_yuv_queue(&b->rgb_to_yuv_state, b->q, rgb_buf->buf_cl, b->yuv_ion[b->cur_yuv_idx].buf_cl);
  visionbuf_sync(&b->yuv_ion[b->cur_yuv_idx], VISIONBUF_SYNC_FROM_DEVICE);

  // keep another reference around till were done processing
  pool_acquire(&b->yuv_pool, b->cur_yuv_idx);
  pool_push(&b->yuv_pool, b->cur_yuv_idx);

  return true;
}

void camerabuf_release(CameraBuf *b) {
  tbuffer_dispatch(&b->ui_tb, b->cur_rgb_idx);
  pool_release(&b->yuv_pool, b->cur_yuv_idx);
}

void camerabuf_stop(CameraBuf *b) {
  tbuffer_stop(&b->ui_tb);
  pool_stop(&b->yuv_pool);
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
  const FrameMetadata &frame_data = b->yuv_metas[b->cur_yuv_idx];
  if (pm != nullptr) {
    MessageBuilder msg;
    auto framed = msg.initEvent().initFrame();
    fill_frame_data(framed, frame_data, cnt);

#if !defined(QCOM) && !defined(QCOM2)
    framed.setImage(kj::arrayPtr((const uint8_t*)b->yuv_ion[b->cur_yuv_idx].addr, b->yuv_buf_size));
#endif
    framed.setTransform(kj::ArrayPtr<const float>(&b->yuv_transform.v[0], 9));
    pm->send("frame", msg);
  }

#ifdef NOSCREEN
  if (frame_data.frame_id % 4 == 1) {
    sendrgb(s, (uint8_t *)b->rgb_bufs[b->cur_rgb_idx].addr, b->rgb_bufs[b->cur_rgb_idx].len, 0);
  }
#endif
}


struct VisionState;

struct VisionClientState {
  VisionState *s;
  int fd;
  pthread_t thread_handle;
  bool running;
};

struct VisionClientStreamState {
  bool subscribed;
  int bufs_outstanding;
  bool tb;
  TBuffer* tbuffer;
  PoolQueue* queue;
};

struct VisionState {
  cl_device_id device_id;
  cl_context context;

  CameraBuf rear;
  CameraBuf front;
#ifdef QCOM2
  CameraBuf wide;
#endif
  
  MultiCameraState cameras;
  zsock_t *terminate_pub;
  PubMaster *pm;
  pthread_mutex_t clients_lock;
  VisionClientState clients[MAX_CLIENTS];
};

static void autoexposure(CameraState *s, uint32_t *lum_binning, int len, int lum_total) {
  unsigned int lum_cur = 0;
  int lum_med = 0;
  for (lum_med = 0; lum_med < len; lum_med++) {
    // shouldn't be any values less than 16 - yuv footroom
    lum_cur += lum_binning[lum_med];
    if (lum_cur >= lum_total / 2) {
      break;
    }
  }
  camera_autoexposure(s, lum_med / 256.0);
}

// frontview thread
void* frontview_thread(VisionState *s) {
  int meteringbox_xmin = 0, meteringbox_xmax = 0;
  int meteringbox_ymin = 0, meteringbox_ymax = 0;
  const bool rhd_front = read_db_bool("IsRHD");

  set_thread_name("frontview");
  set_realtime_priority(51);
  // we subscribe to this for placement of the AE metering box
  // TODO: the loop is bad, ideally models shouldn't affect sensors
  SubMaster sm({"driverState"});

  CameraBuf *b = &s->front;
  for (int cnt = 0; !do_exit; cnt++) {
    if (!camerabuf_acquire(b, &s->cameras.front)) continue;

    if (sm.update(0) > 0) {
      auto state = sm["driverState"].getDriverState();
      // set front camera metering target
      if (state.getFaceProb() > 0.4) {
        auto face_position = state.getFacePosition();

        const int x_offset = rhd_front ? 0 : b->rgb_width - 0.5 * b->rgb_height;
        const int x = x_offset + (face_position[0] + 0.5) * (0.5 * b->rgb_height);
        const int y = (face_position[1] + 0.5) * (b->rgb_height);
        meteringbox_xmin = x - 72;
        meteringbox_xmax = x + 72;
        meteringbox_ymin = y - 72;
        meteringbox_ymax = y + 72;
      } else {// use default setting if no face
        meteringbox_ymin = b->rgb_height * 1 / 3;
        meteringbox_ymax = b->rgb_height * 1;
        meteringbox_xmin = rhd_front ? 0 : b->rgb_width * 3 / 5;
        meteringbox_xmax = rhd_front ? b->rgb_width * 2 / 5 : b->rgb_width;
      }
    }

    // auto exposure
#ifndef DEBUG_DRIVER_MONITOR
    if (cnt % 3 == 0)
#endif
    {
      // use driver face crop for AE
      int x_start, x_end, y_start, y_end;
      int skip = 1;

      if (meteringbox_xmax > 0) {
        x_start = std::max(0, meteringbox_xmin);
        x_end = std::min(b->rgb_width - 1, meteringbox_xmax);
        y_start = std::max(0, meteringbox_ymin);
        y_end = std::min(b->rgb_height - 1, meteringbox_ymax);
      } else {
        y_start = b->rgb_height * 1 / 3;
        y_end = b->rgb_height * 1;
        x_start = rhd_front ? 0 : b->rgb_width * 3 / 5;
        x_end = rhd_front ? b->rgb_width * 2 / 5 : b->rgb_width;
      }
#ifdef QCOM2
      x_start = 0.15 * b->rgb_width;
      x_end = 0.85 * b->rgb_width;
      y_start = 0.5 * b->rgb_height;
      y_end = 0.75 * b->rgb_height;
      skip = 2;
#endif
      const uint8_t *bgr_front_ptr = (const uint8_t*)b->rgb_bufs[b->cur_rgb_idx].addr;
      uint32_t lum_binning[256] = {0,};
      for (int y = y_start; y < y_end; y += skip) {
        for (int x = x_start; x < x_end; x += 2) { // every 2nd col
          const uint8_t *pix = &bgr_front_ptr[y * b->rgb_stride + x * 3];
          unsigned int lum = (unsigned int)pix[0] + pix[1] + pix[2];
#ifdef DEBUG_DRIVER_MONITOR
          uint8_t *pix_rw = (uint8_t *)pix;

          // set all the autoexposure pixels to pure green (pixel format is bgr)
          pix_rw[0] = pix_rw[2] = 0;
          pix_rw[1] = 0xff;
#endif
          lum_binning[std::min(lum / 3, 255u)]++;
        }
      }
      const unsigned int lum_total = (y_end - y_start) * (x_end - x_start) / 2 / skip;
      autoexposure(&s->cameras.front, lum_binning, ARRAYSIZE(lum_binning), lum_total);
    }

    const FrameMetadata &frame_data = b->yuv_metas[b->cur_yuv_idx];
    if (s->pm) {
      MessageBuilder msg;
      auto framed = msg.initEvent().initFrontFrame();
      framed.setFrameType(cereal::FrameData::FrameType::FRONT);
      fill_frame_data(framed, frame_data, cnt);
      s->pm->send("frontFrame", msg);
    }

#ifdef NOSCREEN
    if (frame_data.frame_id % 4 == 2) {
      sendrgb(&s->cameras, (uint8_t *)b->rgb_bufs[b-cur_rgb_idx]->addr, b->rgb_bufs[b-cur_rgb_idx]->len, 2);
    }
#endif

    camerabuf_release(b);
  }
  return NULL;
}

// processing thread
void *processing_thread(VisionState *s, const char *tname, CameraState *camera_state, CameraBuf *b, int priority) {
  set_thread_name(tname);
  int err = set_realtime_priority(priority);
  LOG("setpriority returns %d", err);
  LOG("%s start!", tname);

  for (int cnt = 0; !do_exit; cnt++) {
    if (!camerabuf_acquire(b, camera_state)) continue;

#ifdef QCOM2
    if (camera_state == &s->cameras.wide) {
      camera_wide_process_buf(&s->cameras, b, cnt, s->pm);
    } else
#endif
    {
      camera_process_buf(&s->cameras, b, cnt, s->pm);
    }

    if (cnt % 3 == 0) {
      // auto exposure over big box
#ifdef QCOM2
      const int exposure_x = 384;
      const int exposure_y = 300;
      const int exposure_height = 400;
      const int exposure_width = 1152;
      const int skip = 2;
#else
      const int exposure_x = 290;
      const int exposure_y = 322;
      const int exposure_height = 314;
      const int exposure_width = 560;
      const int skip = 1;
#endif
      uint8_t *yuv_ptr_y = b->yuv_bufs[b->cur_yuv_idx].y;
      // find median box luminance for AE
      uint32_t lum_binning[256] = {0,};
      for (int y = 0; y < exposure_height; y+=skip) {
        for (int x = 0; x < exposure_width; x+=skip) {
          uint8_t lum = yuv_ptr_y[((exposure_y + y) * b->yuv_width) + exposure_x + x];
          lum_binning[lum]++;
        }
      }
      const unsigned int lum_total = exposure_height * exposure_width / skip / skip;
      autoexposure(camera_state, lum_binning, ARRAYSIZE(lum_binning), lum_total);
    }

    camerabuf_release(b);
  }
  return NULL;
}

static CameraBuf *get_camerabuf_by_type(VisionState *s, VisionStreamType type) {
  assert(type >= 0 && type < VISION_STREAM_MAX);
  if (type == VISION_STREAM_RGB_BACK || type == VISION_STREAM_YUV) {
    return &s->rear;
  } else if (type == VISION_STREAM_RGB_FRONT || VISION_STREAM_YUV_FRONT) {
    return &s->front;
  }
#ifdef QCOM2
  else {
    return &s->wide;
  }
#endif
}

// visionserver
void* visionserver_client_thread(void* arg) {
  int err;
  VisionClientState *client = (VisionClientState*)arg;
  VisionState *s = client->s;
  int fd = client->fd;

  set_thread_name("clientthread");

  zsock_t *terminate = zsock_new_sub(">inproc://terminate", "");
  assert(terminate);
  void* terminate_raw = zsock_resolve(terminate);

  VisionClientStreamState streams[VISION_STREAM_MAX] = {{0}};

  LOGW("client start fd %d", fd);

  while (true) {
    zmq_pollitem_t polls[2+VISION_STREAM_MAX] = {{0}};
    polls[0].socket = terminate_raw;
    polls[0].events = ZMQ_POLLIN;
    polls[1].fd = fd;
    polls[1].events = ZMQ_POLLIN;

    int poll_to_stream[2+VISION_STREAM_MAX] = {0};
    int num_polls = 2;
    for (int i=0; i<VISION_STREAM_MAX; i++) {
      if (!streams[i].subscribed) continue;
      polls[num_polls].events = ZMQ_POLLIN;
      if (streams[i].bufs_outstanding >= 2) {
        continue;
      }
      if (streams[i].tb) {
        polls[num_polls].fd = tbuffer_efd(streams[i].tbuffer);
      } else {
        polls[num_polls].fd = poolq_efd(streams[i].queue);
      }
      poll_to_stream[num_polls] = i;
      num_polls++;
    }
    int ret = zmq_poll(polls, num_polls, -1);
    if (ret < 0) {
      if (errno == EINTR || errno == EAGAIN) continue;
      LOGE("poll failed (%d - %d)", ret, errno);
      break;
    }
    if (polls[0].revents) {
      break;
    } else if (polls[1].revents) {
      VisionPacket p;
      err = vipc_recv(fd, &p);
      if (err <= 0) {
        break;
      } else if (p.type == VIPC_STREAM_SUBSCRIBE) {
        VisionStreamType stream_type = p.d.stream_sub.type;
        VisionPacket rep = {
          .type = VIPC_STREAM_BUFS,
          .d = { .stream_bufs = { .type = stream_type }, },
        };

        VisionClientStreamState *stream = &streams[stream_type];
        stream->tb = p.d.stream_sub.tbuffer;

        VisionStreamBufs *stream_bufs = &rep.d.stream_bufs;
        CameraBuf *b = get_camerabuf_by_type(s, stream_type);
        if (stream_type == VISION_STREAM_RGB_BACK ||
            stream_type == VISION_STREAM_RGB_FRONT ||
            stream_type == VISION_STREAM_RGB_WIDE) {
          stream_bufs->width = b->rgb_width;
          stream_bufs->height = b->rgb_height;
          stream_bufs->stride = b->rgb_stride;
          stream_bufs->buf_len = b->rgb_bufs[0].len;
          rep.num_fds = UI_BUF_COUNT;
          for (int i = 0; i < rep.num_fds; i++) {
            rep.fds[i] = b->rgb_bufs[i].fd;
          }
          if (stream->tb) {
            stream->tbuffer = &b->ui_tb;
          } else {
            assert(false);
          }
        } else {
          stream_bufs->width = b->yuv_width;
          stream_bufs->height = b->yuv_height;
          stream_bufs->stride = b->yuv_width;
          stream_bufs->buf_len = b->yuv_buf_size;
          rep.num_fds = YUV_COUNT;
          for (int i = 0; i < rep.num_fds; i++) {
            rep.fds[i] = b->yuv_ion[i].fd;
          }
          if (stream->tb) {
            stream->tbuffer = b->yuv_tb;
          } else {
            stream->queue = pool_get_queue(&b->yuv_pool);
          }
        }
        vipc_send(fd, &rep);
        streams[stream_type].subscribed = true;
      } else if (p.type == VIPC_STREAM_RELEASE) {
        int si = p.d.stream_rel.type;
        assert(si < VISION_STREAM_MAX);
        if (streams[si].tb) {
          tbuffer_release(streams[si].tbuffer, p.d.stream_rel.idx);
        } else {
          poolq_release(streams[si].queue, p.d.stream_rel.idx);
        }
        streams[p.d.stream_rel.type].bufs_outstanding--;
      } else {
        assert(false);
      }
    } else {
      int stream_i = VISION_STREAM_MAX;
      for (int i=2; i<num_polls; i++) {
        int si = poll_to_stream[i];
        if (!streams[si].subscribed) continue;
        if (polls[i].revents) {
          stream_i = si;
          break;
        }
      }
      if (stream_i < VISION_STREAM_MAX) {
        streams[stream_i].bufs_outstanding++;
        int idx;
        if (streams[stream_i].tb) {
          idx = tbuffer_acquire(streams[stream_i].tbuffer);
        } else {
          idx = poolq_pop(streams[stream_i].queue);
        }
        if (idx < 0) {
          break;
        }
        VisionPacket rep = {
          .type = VIPC_STREAM_ACQUIRE,
          .d = {.stream_acq = {
            .type = (VisionStreamType)stream_i,
            .idx = idx,
          }},
        };
        if (stream_i == VISION_STREAM_YUV ||
            stream_i == VISION_STREAM_YUV_FRONT ||
            stream_i == VISION_STREAM_YUV_WIDE) {
          CameraBuf *b = get_camerabuf_by_type(s, (VisionStreamType)stream_i);
          rep.d.stream_acq.extra.frame_id = b->yuv_metas[idx].frame_id;
          rep.d.stream_acq.extra.timestamp_eof = b->yuv_metas[idx].timestamp_eof;
        }
        vipc_send(fd, &rep);
      }
    }
  }

  LOGW("client end fd %d", fd);

  for (int i=0; i<VISION_STREAM_MAX; i++) {
    if (!streams[i].subscribed) continue;
    if (streams[i].tb) {
      tbuffer_release_all(streams[i].tbuffer);
    } else {
      pool_release_queue(streams[i].queue);
    }
  }

  close(fd);
  zsock_destroy(&terminate);

  pthread_mutex_lock(&s->clients_lock);
  client->running = false;
  pthread_mutex_unlock(&s->clients_lock);

  return NULL;
}

void* visionserver_thread(void* arg) {
  int err;
  VisionState *s = (VisionState*)arg;

  set_thread_name("visionserver");

  zsock_t *terminate = zsock_new_sub(">inproc://terminate", "");
  assert(terminate);
  void* terminate_raw = zsock_resolve(terminate);

  int sock = ipc_bind(VIPC_SOCKET_PATH);
  while (!do_exit) {
    zmq_pollitem_t polls[2] = {{0}};
    polls[0].socket = terminate_raw;
    polls[0].events = ZMQ_POLLIN;
    polls[1].fd = sock;
    polls[1].events = ZMQ_POLLIN;

    int ret = zmq_poll(polls, ARRAYSIZE(polls), -1);
    if (ret < 0) {
      if (errno == EINTR || errno == EAGAIN) continue;
      LOGE("poll failed (%d - %d)", ret, errno);
      break;
    }
    if (polls[0].revents) {
      break;
    } else if (!polls[1].revents) {
      continue;
    }

    int fd = accept(sock, NULL, NULL);
    assert(fd >= 0);

    pthread_mutex_lock(&s->clients_lock);

    int client_idx = 0;
    for (; client_idx < MAX_CLIENTS; client_idx++) {
      if (!s->clients[client_idx].running) break;
    }

    if (client_idx >= MAX_CLIENTS) {
      LOG("ignoring visionserver connection, max clients connected");
      close(fd);

      pthread_mutex_unlock(&s->clients_lock);
      continue;
    }

    VisionClientState *client = &s->clients[client_idx];
    client->s = s;
    client->fd = fd;
    client->running = true;

    err = pthread_create(&client->thread_handle, NULL,
                         visionserver_client_thread, client);
    assert(err == 0);

    pthread_mutex_unlock(&s->clients_lock);
  }

  for (int i=0; i<MAX_CLIENTS; i++) {
    pthread_mutex_lock(&s->clients_lock);
    bool running = s->clients[i].running;
    pthread_mutex_unlock(&s->clients_lock);
    if (running) {
      err = pthread_join(s->clients[i].thread_handle, NULL);
      assert(err == 0);
    }
  }

  close(sock);
  zsock_destroy(&terminate);

  return NULL;
}

void cl_init(VisionState *s) {
  int err;
  clu_init();
  s->device_id = cl_get_device_id(CL_DEVICE_TYPE_DEFAULT);
  s->context = clCreateContext(NULL, 1, &s->device_id, NULL, NULL, &err);
  assert(err == 0);
}

void init_buffers(VisionState *s) {
  camerabuf_init(&s->rear, s->device_id, s->context, &s->cameras.rear, "rgb");
  camerabuf_init(&s->front, s->device_id, s->context, &s->cameras.front, "frontrgb");
#ifdef QCOM2
  camerabuf_init(&s->wide, s->device_id, s->context, &s->cameras.wide, "widergb");
#endif
}

void free_buffers(VisionState *s) {
  camerabuf_free(&s->rear);
  camerabuf_free(&s->front);
#ifdef QCOM2
  camerabuf_free(&s->wide);
#endif
}

void party(VisionState *s) {
  s->terminate_pub = zsock_new_pub("@inproc://terminate");
  assert(s->terminate_pub);

  std::vector<std::thread> threads;
  threads.push_back(std::thread(visionserver_thread, s));
  threads.push_back(std::thread(processing_thread, s, "processing", &s->cameras.rear, &s->rear, 51));
#ifndef __APPLE__
  threads.push_back(std::thread(frontview_thread, s));
#endif
#ifdef QCOM2
  threads.push_back(std::thread(processing_thread, s, "wideview", &s->cameras.wide, &s->wide, 51));
#endif

  // priority for cameras
  int err = set_realtime_priority(51);
  LOG("setpriority returns %d", err);

  cameras_run(&s->cameras);

  camerabuf_stop(&s->rear);
  camerabuf_stop(&s->front);
#ifdef QCOM2
  camerabuf_stop(&s->wide);
#endif
  zsock_signal(s->terminate_pub, 0);

  for (auto &t : threads) t.join();

  zsock_destroy(&s->terminate_pub);
}

int main(int argc, char *argv[]) {
  set_realtime_priority(51);
#ifdef QCOM
  set_core_affinity(2);
#endif

  zsys_handler_set(NULL);
  signal(SIGINT, (sighandler_t)set_do_exit);
  signal(SIGTERM, (sighandler_t)set_do_exit);

  VisionState state = {};
  VisionState *s = &state;

  cl_init(s);

  cameras_init(&s->cameras);

  init_buffers(s);

#if defined(QCOM) && !defined(QCOM_REPLAY)
  s->pm = new PubMaster({"frame", "frontFrame"});
#elif defined(QCOM2)
  s->pm = new PubMaster({"frame", "frontFrame", "wideFrame"});
#endif

#ifndef QCOM2
  cameras_open(&s->cameras, &s->rear.camera_bufs[0], &s->front.camera_bufs[0]);
#else
  cameras_open(&s->cameras, &s->rear.camera_bufs[0], &s->front.camera_bufs[0], &s->wide.camera_bufs[0]);
#endif

  party(s);

  if (s->pm != NULL) {
    delete s->pm;
  }

  free_buffers(s);
  clReleaseContext(s->context);
}
