#include "tools/replay/camera.h"
#include "tools/replay/util.h"

#include <cassert>

CameraServer::CameraServer(std::pair<int, int> camera_size[MAX_CAMERAS]) {
  for (int i = 0; i < MAX_CAMERAS; ++i) {
    std::tie(cameras_[i].width, cameras_[i].height) = camera_size[i];
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
      rInfo("camera[%d] frame size %dx%d", cam.type, cam.width, cam.height);
      vipc_server_->create_buffers(cam.stream_type, YUV_BUFFER_COUNT, false, cam.width, cam.height);
      if (!cam.thread.joinable()) {
        cam.thread = std::thread(&CameraServer::cameraThread, this, std::ref(cam));
      }
    }
  }
  vipc_server_->start_listener();
}

void CameraServer::cameraThread(Camera &cam) {
  auto read_frame = [&](FrameReader *fr, int frame_id) {
    VisionBuf *yuv_buf = vipc_server_->get_buffer(cam.stream_type);
    assert(yuv_buf);
    bool ret = fr->get(frame_id, (uint8_t *)yuv_buf->addr);
    return ret ? yuv_buf : nullptr;
  };

  while (true) {
    const auto [fr, eidx] = cam.queue.pop();
    if (!fr) break;

    const int id = eidx.getSegmentId();
    bool prefetched = (id == cam.cached_id && eidx.getSegmentNum() == cam.cached_seg);
    auto yuv = prefetched ? cam.cached_buf : read_frame(fr, id);
    if (yuv) {
      VisionIpcBufExtra extra = {
          .frame_id = eidx.getFrameId(),
          .timestamp_sof = eidx.getTimestampSof(),
          .timestamp_eof = eidx.getTimestampEof(),
      };
      yuv->set_frame_id(eidx.getFrameId());
      vipc_server_->send(yuv, &extra);
    } else {
      rError("camera[%d] failed to get frame: %lu", cam.type, eidx.getSegmentId());
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
    waitForSent();
    startVipcServer();
  }

  ++publishing_;
  cam.queue.push({fr, eidx});
}

void CameraServer::waitForSent() {
  while (publishing_ > 0) {
    std::this_thread::yield();
  }
}
