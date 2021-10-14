#include "selfdrive/ui/replay/camera.h"

#include <cassert>
#include <iostream>

const int YUV_BUF_COUNT = 50;

CameraServer::CameraServer() {}

CameraServer::~CameraServer() {
  // exit all camera threads
  for (auto &cam : cameras_) {
    if (cam.thread.joinable()) {
      cam.queue.push({});
      cam.thread.join();
    }
  }
  vipc_server_.reset(nullptr);
}

void CameraServer::start(std::pair<int, int> camera_size[MAX_CAMERAS]) {
  for (auto type : ALL_CAMERAS) {
    std::tie(cameras_[type].width, cameras_[type].height) = camera_size[type];
  }
  startVipcServer();
}

void CameraServer::startVipcServer() {
  vipc_server_.reset(new VisionIpcServer("camerad"));
  for (auto &type : ALL_CAMERAS) {
    auto &cam = cameras_[type];
    if (cam.width > 0 && cam.height > 0) {
      vipc_server_->create_buffers(cam.rgb_type, UI_BUF_COUNT, true, cam.width, cam.height);
      vipc_server_->create_buffers(cam.yuv_type, YUV_BUF_COUNT, false, cam.width, cam.height);
      // Create thread on demand to handle new incoming camera
      if (!cam.thread.joinable()) {
        cam.thread = std::thread(&CameraServer::cameraThread, this, type, std::ref(cam));
      }
    }
  }
  vipc_server_->start_listener();
}

void CameraServer::cameraThread(CameraType type, Camera &cam) {
  while (true) {
    const auto [fr, eidx] = cam.queue.pop();
    if (!fr) break;

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
  auto &cam = cameras_[type];
  if (cam.width != fr->width || cam.height != fr->height) {
    cam.width = fr->width;
    cam.height = fr->height;
    std::cout << "camera[" << type << "] frame size " << cam.width << "x" << cam.height << std::endl;
    // wait until all cameras have finished publishing before restart vipc server.
    waitFinish();
    startVipcServer();
  }

  ++publishing_;
  cam.queue.push({fr, eidx});
}
