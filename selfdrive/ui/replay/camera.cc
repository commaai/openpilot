#include "selfdrive/ui/replay/camera.h"

#include <cassert>
#include <iostream>

static const VisionStreamType stream_types[] = {
    [RoadCam] = VISION_STREAM_RGB_BACK,
    [DriverCam] = VISION_STREAM_RGB_FRONT,
    [WideRoadCam] = VISION_STREAM_RGB_WIDE,
};

CameraServer::CameraServer() {
  device_id_ = cl_get_device_id(CL_DEVICE_TYPE_DEFAULT);
  context_ = CL_CHECK_ERR(clCreateContext(NULL, 1, &device_id_, NULL, NULL, &err));

  camera_thread_ = std::thread(&CameraServer::thread, this);
  start();
}

CameraServer::~CameraServer() {
  // stop camera thread
  queue_.push({});
  camera_thread_.join();

  delete vipc_server_;
  CL_CHECK(clReleaseContext(context_));
}

void CameraServer::start() {
  vipc_server_ = new VisionIpcServer("camerad", device_id_, context_);
  vipc_server_->start_listener();
}

void CameraServer::thread() {
  while (true) {
    const auto [cam_type, fr, encodeId] = queue_.pop();
    if (!fr) break;

    Camera &cam = cameras_[cam_type];
    if (cam.width != fr->width || cam.height != fr->height) {
      bool buffer_initialized = cam.width > 0 && cam.height > 0;
      cam.width = fr->width;
      cam.height = fr->height;
      if (!buffer_initialized) {
        vipc_server_->create_buffers(stream_types[cam_type], UI_BUF_COUNT, true, cam.width, cam.height);
      } else {
        std::cout << "frame size changed, restart vipc server" << std::endl;
        delete vipc_server_;
        cameras_.fill({});
        start();
        continue;
      }
    }

    if (uint8_t *dat = fr->get(encodeId)) {
      VisionIpcBufExtra extra = {};
      VisionBuf *buf = vipc_server_->get_buffer(stream_types[cam_type]);
      memcpy(buf->addr, dat, fr->getRGBSize());
      vipc_server_->send(buf, &extra, false);
    } else {
      std::cout << "failed get frame. camera:" << cam_type << ", encodeId:" << encodeId << std::endl;
    }
  }
}
