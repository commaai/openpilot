#include "selfdrive/ui/replay/camera.h"

#include <cassert>
#include <iostream>

const int YUV_BUF_COUNT = 50;

CameraServer::CameraServer() {
  camera_thread_ = std::thread(&CameraServer::thread, this);
}

CameraServer::~CameraServer() {
  queue_.push({});
  camera_thread_.join();
  vipc_server_.reset(nullptr);
}

void CameraServer::start(std::pair<int, int> cameras[MAX_CAMERAS]) {
  for (auto type : ALL_CAMERAS) {
    std::tie(cameras_[type].width, cameras_[type].height) = cameras[type];
  }
  startVipcServer();
}

void CameraServer::startVipcServer() {
  std::cout << (vipc_server_ ? "restart" : "start") << " vipc server" << std::endl;
  vipc_server_.reset(new VisionIpcServer("camerad"));
  for (auto &cam : cameras_) {
    if (cam.width > 0 && cam.height > 0) {
      vipc_server_->create_buffers(cam.rgb_type, UI_BUF_COUNT, true, cam.width, cam.height);
      vipc_server_->create_buffers(cam.yuv_type, YUV_BUF_COUNT, false, cam.width, cam.height);
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
    if (auto dat = fr->get(eidx.getSegmentId())) {
      auto [rgb_dat, yuv_dat] = *dat;
      VisionIpcBufExtra extra = {
          .frame_id = eidx.getFrameId(),
          .timestamp_sof = eidx.getTimestampSof(),
          .timestamp_eof = eidx.getTimestampEof(),
      };

      VisionBuf *rgb_buf = vipc_server_->get_buffer(cam.rgb_type);
      memcpy(rgb_buf->addr, rgb_dat, fr->getRGBSize());
      VisionBuf *yuv_buf = vipc_server_->get_buffer(cam.yuv_type);
      memcpy(yuv_buf->addr, yuv_dat, fr->getYUVSize());

      vipc_server_->send(rgb_buf, &extra, false);
      vipc_server_->send(yuv_buf, &extra, false);
    } else {
      std::cout << "camera[" << type << "] failed to get frame:" << eidx.getSegmentId() << std::endl;
    }

    --publishing_;
  }
}

void CameraServer::pushFrame(CameraType type, FrameReader *fr, const cereal::EncodeIndex::Reader &eidx) {
  ++publishing_;
  queue_.push({type, fr, eidx});
}
