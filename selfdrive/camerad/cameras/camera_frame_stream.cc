#include "camera_frame_stream.h"

#include <unistd.h>
#include <cassert>

#include <capnp/dynamic.h>

#include "messaging.hpp"
#include "common/util.h"

#define FRAME_WIDTH 1164
#define FRAME_HEIGHT 874

extern ExitHandler do_exit;

namespace {

// TODO: make this more generic
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
    .bayer = false,
    .bayer_flip = 3,
    .hdr = false
  },
};

void camera_init(CameraServer *server, CameraState *s, int camera_id, unsigned int fps) {
  assert(camera_id < ARRAYSIZE(cameras_supported));
  s->ci = cameras_supported[camera_id];
  assert(s->ci.frame_width != 0);

  s->camera_num = camera_id;
  s->fps = fps;
  s->buf.init(server, s, FRAME_BUF_COUNT);
}

void run_frame_stream(CameraState &camera, const char* frame_pkt) {
  SubMaster sm({frame_pkt});

  size_t buf_idx = 0;
  while (!do_exit) {
    if (sm.update(1000) == 0) continue;

    auto msg = static_cast<capnp::DynamicStruct::Reader>(sm[frame_pkt]);
    auto frame = msg.get(frame_pkt).as<capnp::DynamicStruct>();
    camera.buf.camera_bufs_metadata[buf_idx] = {
      .frame_id = frame.get("frameId").as<uint32_t>(),
      .timestamp_eof = frame.get("timestampEof").as<uint64_t>(),
      .frame_length = frame.get("frameLength").as<unsigned>(),
      .integ_lines = frame.get("integLines").as<unsigned>(),
      .global_gain = frame.get("globalGain").as<unsigned>(),
    };

    cl_command_queue q = camera.buf.camera_bufs[buf_idx].copy_q;
    cl_mem yuv_cl = camera.buf.camera_bufs[buf_idx].buf_cl;

    auto image = frame.get("image").as<capnp::Data>();
    clEnqueueWriteBuffer(q, yuv_cl, CL_TRUE, 0, image.size(), image.begin(), 0, NULL, NULL);
    camera.buf.queue(buf_idx);
    buf_idx = (buf_idx + 1) % FRAME_BUF_COUNT;
  }
}

}  // namespace

void camera_autoexposure(CameraState *s, float grey_frac) {}
void process_road_camera(CameraServer *s, CameraState *c, int cnt) {}

// CameraServer

CameraServer::CameraServer() : CameraServerBase() {
  camera_init(this, &road_cam, CAMERA_ID_IMX298, 20);
  camera_init(this, &driver_cam, CAMERA_ID_OV8865, 10);
}

void CameraServer::run() {
  start_process_thread(&road_cam, process_road_camera);
  set_thread_name("frame_streaming");
  run_frame_stream(road_cam, "roadCameraState");
}
