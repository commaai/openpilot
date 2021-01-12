#include "camera_frame_stream.h"

#include <unistd.h>
#include <cassert>
#include <string.h>

#include <libyuv.h>
#include "messaging.hpp"

#include "common/util.h"
#include "common/timing.h"
#include "common/swaglog.h"

extern "C" {
#include <libavcodec/avcodec.h>
}

extern ExitHandler do_exit;

#define FRAME_WIDTH 1164
#define FRAME_HEIGHT 874

namespace {
void camera_open(CameraState *s, bool rear) {
}

void camera_close(CameraState *s) {
  // empty
}

void camera_init(VisionIpcServer * v, CameraState *s, int camera_id, unsigned int fps, cl_device_id device_id, cl_context ctx, VisionStreamType rgb_type, VisionStreamType yuv_type) {
  assert(camera_id < ARRAYSIZE(cameras_supported));
  s->ci = cameras_supported[camera_id];
  assert(s->ci.frame_width != 0);

  s->fps = fps;
  s->buf.init(device_id, ctx, s, v, FRAME_BUF_COUNT, rgb_type, yuv_type);
}

void run_frame_stream(MultiCameraState *s) {
  s->sm = new SubMaster({"frame"});

  CameraState *const rear_camera = &s->rear;


  size_t buf_idx = 0;
  while (!do_exit) {
    if (s->sm->update(1000) == 0) continue;

    auto frame = (*(s->sm))["frame"].getFrame();
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
    rear_camera->buf.queue(buf_idx);
    buf_idx = (buf_idx + 1) % FRAME_BUF_COUNT;
  }

  delete s->sm;
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

void cameras_init(VisionIpcServer *v, MultiCameraState *s, cl_device_id device_id, cl_context ctx) {
  camera_init(v, &s->rear, CAMERA_ID_IMX298, 20, device_id, ctx,
              VISION_STREAM_RGB_BACK, VISION_STREAM_YUV_BACK);
  camera_init(v, &s->front, CAMERA_ID_OV8865, 10, device_id, ctx,
              VISION_STREAM_RGB_FRONT, VISION_STREAM_YUV_FRONT);
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
  // empty
}

void cameras_run(MultiCameraState *s) {
  std::thread t = start_process_thread(s, "processing", &s->rear, camera_process_rear);
  set_thread_name("frame_streaming");
  run_frame_stream(s);
  t.join();
  cameras_close(s);
}
