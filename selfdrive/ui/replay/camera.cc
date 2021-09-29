#include "selfdrive/ui/replay/camera.h"

#include <cassert>
#include <iostream>

// TODO: publish YUV frame
// const int YUV_BUF_COUNT = 50;

CameraServer::CameraServer() {
  device_id_ = cl_get_device_id(CL_DEVICE_TYPE_DEFAULT);
  context_ = CL_CHECK_ERR(clCreateContext(nullptr, 1, &device_id_, nullptr, nullptr, &err));
  camera_thread_ = std::thread(&CameraServer::thread, this);
}

CameraServer::~CameraServer() {
  queue_.push({});
  camera_thread_.join();
  vipc_server_.reset(nullptr);
  CL_CHECK(clReleaseContext(context_));
}

void CameraServer::startVipcServer() {
  std::cout << (vipc_server_ ? "start" : "restart") << " vipc server" << std::endl;
  vipc_server_.reset(new VisionIpcServer("camerad", device_id_, context_));
  for (auto &cam : cameras_) {
    if (cam.width > 0 && cam.height > 0) {
      vipc_server_->create_buffers(cam.rgb_type, UI_BUF_COUNT, true, cam.width, cam.height);
      // vipc_server_->create_buffers(cam.yuv_type, YUV_BUF_COUNT, false, cam.width, cam.height);
    }
  }
  vipc_server_->start_listener();
}

void CameraServer::thread() {
  while (true) {
    const auto [type, fr, eidx] = queue_.pop();
    if (!fr) break;

    auto &cam = cameras_[type];
    // start|restart the vipc server if frame size changed
    if (cam.width != fr->width || cam.height != fr->height) {
      cam.width = fr->width;
      cam.height = fr->height;
      std::cout << "camera[" << type << "] frame size " << cam.width << "x" << cam.height << std::endl;
      startVipcServer();
    }

    // send frame
    if (uint8_t *dat = fr->get(eidx.getSegmentId())) {
      VisionIpcBufExtra extra = {
          .frame_id = eidx.getFrameId(),
          .timestamp_sof = eidx.getTimestampSof(),
          .timestamp_eof = eidx.getTimestampEof(),
      };

      VisionBuf *rgb_buf = vipc_server_->get_buffer(cam.rgb_type);
      memcpy(rgb_buf->addr, dat, fr->getRGBSize());
      vipc_server_->send(rgb_buf, &extra, false);
    } else {
      std::cout << "camera[" << type << "] failed to get frame:" << eidx.getSegmentId() << std::endl;
    }
  }
}
