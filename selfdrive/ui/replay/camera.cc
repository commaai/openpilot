#include "selfdrive/ui/replay/camera.h"

#include <cassert>
#include <iostream>

const int YUV_BUF_COUNT = 50;

CameraServer::CameraServer(std::pair<int, int> camera_size[MAX_CAMERAS], bool yuv, bool rgb) : send_yuv_(yuv), send_rgb_(rgb) {
  assert(send_yuv_ || send_rgb_);
  for (auto &cam : cameras_) {
    std::tie(cam.width, cam.height) = camera_size[cam.type];
  }
  startVipcServer();
}

CameraServer::~CameraServer() {
  for (auto &cam : cameras_) {
    if (cam.thread.joinable()) {
      cam.queue.push({});
      cam.thread.join();
    }
  }
  vipc_server_.reset(nullptr);
}

void CameraServer::startVipcServer() {
  vipc_server_.reset(new VisionIpcServer("camerad"));
  for (auto &cam : cameras_) {
    if (cam.width > 0 && cam.height > 0) {
      std::cout << "camera[" << cam.type << "] frame size " << cam.width << "x" << cam.height << std::endl;
      if (send_rgb_) vipc_server_->create_buffers(cam.rgb_type, UI_BUF_COUNT, true, cam.width, cam.height);
      if (send_yuv_) vipc_server_->create_buffers(cam.yuv_type, YUV_BUF_COUNT, false, cam.width, cam.height);
      if (!cam.thread.joinable()) {
        cam.thread = std::thread(&CameraServer::cameraThread, this, std::ref(cam));
      }
    }
  }
  vipc_server_->start_listener();
}

void CameraServer::cameraThread(Camera &cam) {
  auto read_frame = [&](FrameReader *fr, int frame_id) {
    VisionBuf *rgb_buf = send_rgb_ ? vipc_server_->get_buffer(cam.rgb_type) : nullptr;
    VisionBuf *yuv_buf = send_yuv_ ? vipc_server_->get_buffer(cam.yuv_type) : nullptr;
    bool ret = fr->get(frame_id, (uint8_t *)rgb_buf->addr, yuv_buf ? (uint8_t *)yuv_buf->addr : nullptr);
    return ret ? std::pair{rgb_buf, yuv_buf} : std::pair{nullptr, nullptr};
  };

  while (true) {
    const auto [fr, eidx] = cam.queue.pop();
    if (!fr) break;

    const int id = eidx.getSegmentId();
    bool prefetched = (id == cam.cached_id && eidx.getSegmentNum() == cam.cached_seg && (cam.cached_buf.first || cam.cached_buf.second));
    auto [rgb, yuv] = prefetched ? cam.cached_buf : read_frame(fr, id);

    if (rgb || yuv) {
      VisionIpcBufExtra extra = {
          .frame_id = eidx.getFrameId(),
          .timestamp_sof = eidx.getTimestampSof(),
          .timestamp_eof = eidx.getTimestampEof(),
      };
      if (send_rgb_) vipc_server_->send(rgb, &extra, false);
      if (send_yuv_) vipc_server_->send(yuv, &extra, false);
    } else {
      std::cout << "camera[" << cam.type << "] failed to get frame:" << eidx.getSegmentId() << std::endl;
    }

    cam.cached_id = id + 1;
    cam.cached_seg = eidx.getSegmentNum();
    cam.cached_buf = read_frame(fr, cam.cached_id);

    --publishing_;
  }
}

void CameraServer::pushFrame(CameraType type, FrameReader *fr, const cereal::EncodeIndex::Reader &eidx) {
  auto &cam = cameras_[type];
  if (cam.width != fr->width || cam.height != fr->height) {
    cam.width = fr->width;
    cam.height = fr->height;
    waitFinish();
    startVipcServer();
  }

  ++publishing_;
  cam.queue.push({fr, eidx});
}
