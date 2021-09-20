#include "selfdrive/ui/replay/camera.h"

#include <cassert>
#include <iostream>

const int YUV_BUF_COUNT = 50;

CameraServer::CameraServer() {
  device_id_ = cl_get_device_id(CL_DEVICE_TYPE_DEFAULT);
  context_ = CL_CHECK_ERR(clCreateContext(NULL, 1, &device_id_, NULL, NULL, &err));
}

CameraServer::~CameraServer() {
  stop();
  CL_CHECK(clReleaseContext(context_));
}

void CameraServer::start() {
  vipc_server_ = new VisionIpcServer("camerad", device_id_, context_);
  for (auto &cam : cameras_) {
    if (cam.width > 0 && cam.height > 0) {
      vipc_server_->create_buffers(cam.rgb_type, UI_BUF_COUNT, true, cam.width, cam.height);
      vipc_server_->create_buffers(cam.yuv_type, YUV_BUF_COUNT, false, cam.width, cam.height);
    }
  }
  camera_thread_ = std::thread(&CameraServer::thread, this);
  vipc_server_->start_listener();
}

void CameraServer::stop() {
  if (vipc_server_) {
    queue_.push({});
    camera_thread_.join();
    delete vipc_server_;
    vipc_server_ = nullptr;
  }
}

void CameraServer::pushFrame(CameraType type, FrameReader *fr, uint32_t encodeFrameId, const cereal::FrameData::Reader &frame_data) {
  auto &cam = cameras_[type];
  if (cam.width != fr->width || cam.height != fr->height) {
    cam.width = fr->width;
    cam.height = fr->height;
    std::cout << "camera["<< type << "] frame changed, restart vipc server" << std::endl;
    stop();
    start();
  }
  queue_.push({type, fr, encodeFrameId, frame_data});
}

void CameraServer::thread() {
  while (true) {
    const auto [type, fr, encodeId, frame_data] = queue_.pop();
    if (!fr) break;

    auto &cam = cameras_[type];
    if (auto dat = fr->get(encodeId)) {
      auto [rgb_dat, yuv_dat] = *dat;
      VisionIpcBufExtra extra = {
          frame_data.getFrameId(),
          frame_data.getTimestampSof(),
          frame_data.getTimestampEof(),
      };

      VisionBuf *rgb_buf = vipc_server_->get_buffer(cam.rgb_type);
      memcpy(rgb_buf->addr, rgb_dat, fr->getRGBSize());
      vipc_server_->send(rgb_buf, &extra, false);

      VisionBuf *yuv_buf = vipc_server_->get_buffer(cam.yuv_type);
      memcpy(yuv_buf->addr, yuv_dat, fr->getYUVSize());
      vipc_server_->send(yuv_buf, &extra, false);
    } else {
      std::cout << "failed get frame. camera:" << type << ", encodeId:" << encodeId << std::endl;
    }
  }
}

void CameraServer::waitFramesSent() {
  while (!queue_.empty()) {
    usleep(0);
  }
}
