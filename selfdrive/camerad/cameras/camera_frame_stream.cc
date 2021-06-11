#include "selfdrive/camerad/cameras/camera_frame_stream.h"

#include <unistd.h>
#include <cassert>

#include <capnp/dynamic.h>

#include "cereal/messaging/messaging.h"
#include "selfdrive/common/util.h"

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

void camera_init(VisionIpcServer * v, CameraState *s, int camera_id, unsigned int fps, cl_device_id device_id, cl_context ctx, VisionStreamType rgb_type, VisionStreamType yuv_type) {
  assert(camera_id < std::size(cameras_supported));
  s->ci = cameras_supported[camera_id];
  assert(s->ci.frame_width != 0);

  s->camera_num = camera_id;
  s->fps = fps;
  s->buf.init(device_id, ctx, s, v, FRAME_BUF_COUNT, rgb_type, yuv_type);
}

void run_frame_stream(CameraState &camera, const char* frame_pkt) {
  SubMaster sm({frame_pkt});

  size_t buf_idx = 0;
  while (!do_exit) {
    sm.update(1000);
    if(sm.updated(frame_pkt)) {
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
}

}  // namespace

void cameras_init(VisionIpcServer *v, MultiCameraState *s, cl_device_id device_id, cl_context ctx) {
  camera_init(v, &s->road_cam, CAMERA_ID_IMX298, 20, device_id, ctx,
              VISION_STREAM_RGB_BACK, VISION_STREAM_YUV_BACK);
  camera_init(v, &s->driver_cam, CAMERA_ID_OV8865, 10, device_id, ctx,
              VISION_STREAM_RGB_FRONT, VISION_STREAM_YUV_FRONT);
}

void cameras_open(MultiCameraState *s) {}
void cameras_close(MultiCameraState *s) {}
void camera_autoexposure(CameraState *s, float grey_frac) {}
void process_road_camera(MultiCameraState *s, CameraState *c, int cnt) {}

void cameras_run(MultiCameraState *s) {
  std::thread t = start_process_thread(s, &s->road_cam, process_road_camera);
  set_thread_name("frame_streaming");
  run_frame_stream(s->road_cam, "roadCameraState");
  t.join();
}
