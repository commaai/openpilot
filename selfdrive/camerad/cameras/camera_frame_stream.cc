#include "camera_frame_stream.h"

#include <unistd.h>
#include <cassert>
#include <string.h>
#include <signal.h>

#include <libyuv.h>
#include "messaging.hpp"

#include "common/util.h"
#include "common/timing.h"
#include "common/swaglog.h"
#include "buffering.h"

extern "C" {
#include <libavcodec/avcodec.h>
}

extern volatile sig_atomic_t do_exit;

#define FRAME_WIDTH 1164
#define FRAME_HEIGHT 874

namespace {
void camera_open(CameraState *s, bool rear) {
}

void camera_close(CameraState *s) {
  s->buf.stop();
}

void camera_init(CameraState *s, int camera_id, unsigned int fps, cl_device_id device_id, cl_context ctx) {
  assert(camera_id < ARRAYSIZE(cameras_supported));
  s->ci = cameras_supported[camera_id];
  assert(s->ci.frame_width != 0);

  s->fps = fps;
  s->buf.init(device_id, ctx, s, FRAME_BUF_COUNT, "camera");
}

void run_frame_stream(MultiCameraState *s) {
  s->sm = new SubMaster({"frame"});

  CameraState *const rear_camera = &s->rear;
  auto *tb = &rear_camera->buf.camera_tb;

  while (!do_exit) {
    if (s->sm->update(1000) == 0) continue;

    auto frame = (*(s->sm))["frame"].getFrame();

    const int buf_idx = tbuffer_select(tb);
    rear_camera->buf.camera_bufs_metadata[buf_idx] = {
      .frame_id = frame.getFrameId(),
      .timestamp_eof = frame.getTimestampEof(),
      .frame_length = static_cast<unsigned>(frame.getFrameLength()),
      .integ_lines = static_cast<unsigned>(frame.getIntegLines()),
      .global_gain = static_cast<unsigned>(frame.getGlobalGain()),
    };

    cl_command_queue q = rear_camera->buf.camera_bufs[buf_idx].copy_q;
    cl_mem yuv_cl = rear_camera->buf.camera_bufs[buf_idx].buf_cl;

    clEnqueueWriteBuffer(q, yuv_cl, CL_TRUE, 0, frame.getImage().size(), frame.getImage().begin(), 0, NULL, NULL);
    tbuffer_dispatch(tb, buf_idx);
  }
}

}  // namespace

CameraInfo cameras_supported[CAMERA_ID_MAX] = {
  [CAMERA_ID_IMX298] = {
    .frame_width = FRAME_WIDTH,
    .frame_height = FRAME_HEIGHT,
    .frame_stride = FRAME_WIDTH*3,
    .bayer = false,
    .bayer_flip = false,
  },
  [CAMERA_ID_OV8865] = {
    .frame_width = 1632,
    .frame_height = 1224,
    .frame_stride = 2040, // seems right
    .bayer = true,
    .bayer_flip = 3,
    .hdr = false
  },
};

void cameras_init(MultiCameraState *s, cl_device_id device_id, cl_context ctx) {
  camera_init(&s->rear, CAMERA_ID_IMX298, 20, device_id, ctx);
  s->rear.transform = (mat3){{
    1.0,  0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0,  0.0, 1.0,
  }};

  camera_init(&s->front, CAMERA_ID_OV8865, 10, device_id, ctx);
  s->front.transform = (mat3){{
    1.0,  0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0,  0.0, 1.0,
  }};
}

void camera_autoexposure(CameraState *s, float grey_frac) {}

void cameras_open(MultiCameraState *s) {
  // LOG("*** open front ***");
  camera_open(&s->front, false);

  // LOG("*** open rear ***");
  camera_open(&s->rear, true);
}

void cameras_close(MultiCameraState *s) {
  camera_close(&s->rear);
}

// called by processing_thread
void camera_process_rear(MultiCameraState *s, CameraState *c, int cnt) {
  if (cnt % 100 == 3) {
    const CameraBuf *b = &c->buf;
    create_thumbnail(s, c, (uint8_t*)b->cur_rgb_buf->addr);
  }
}

void cameras_run(MultiCameraState *s) {
  std::thread t = start_process_thread(s, "processing", &s->rear, camera_process_rear);
  set_thread_name("frame_streaming");
  run_frame_stream(s);
  cameras_close(s);
  t.join();
}
