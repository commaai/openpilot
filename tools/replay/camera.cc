#include "tools/replay/camera.h"

#include <cassert>
#include <algorithm>

#include <capnp/dynamic.h>

#include "third_party/linux/include/msm_media_info.h"
#include "tools/replay/util.h"

const int BUFFER_COUNT = 40;

std::tuple<size_t, size_t, size_t> get_nv12_info(int width, int height) {
  int nv12_width = VENUS_Y_STRIDE(COLOR_FMT_NV12, width);
  int nv12_height = VENUS_Y_SCANLINES(COLOR_FMT_NV12, height);
  assert(nv12_width == VENUS_UV_STRIDE(COLOR_FMT_NV12, width));
  assert(nv12_height / 2 == VENUS_UV_SCANLINES(COLOR_FMT_NV12, height));
  size_t nv12_buffer_size = 2346 * nv12_width;  // comes from v4l2_format.fmt.pix_mp.plane_fmt[0].sizeimage
  return {nv12_width, nv12_height, nv12_buffer_size};
}

CameraServer::CameraServer(std::pair<int, int> camera_size[MAX_CAMERAS]) {
  for (int i = 0; i < MAX_CAMERAS; ++i) {
    std::tie(cameras_[i].width, cameras_[i].height) = camera_size[i];
  }
  startVipcServer();
}

CameraServer::~CameraServer() {
  for (auto &cam : cameras_) {
    if (cam.thread.joinable()) {
      // Clear the queue
      std::pair<FrameReader*, const Event *> item;
      while (cam.queue.try_pop(item)) {
        --publishing_;
      }

      // Signal termination and join the thread
      cam.queue.push({});
      cam.thread.join();
    }
  }
  vipc_server_.reset(nullptr);
}

void CameraServer::startVipcServer() {
  vipc_server_.reset(new VisionIpcServer("camerad"));
  for (auto &cam : cameras_) {
    cam.cached_buf.clear();

    if (cam.width > 0 && cam.height > 0) {
      rInfo("camera[%d] frame size %dx%d", cam.type, cam.width, cam.height);
      auto [nv12_width, nv12_height, nv12_buffer_size] = get_nv12_info(cam.width, cam.height);
      vipc_server_->create_buffers_with_sizes(cam.stream_type, BUFFER_COUNT, cam.width, cam.height,
                                              nv12_buffer_size, nv12_width, nv12_width * nv12_height);
      if (!cam.thread.joinable()) {
        cam.thread = std::thread(&CameraServer::cameraThread, this, std::ref(cam));
      }
    }
  }
  vipc_server_->start_listener();
}

void CameraServer::cameraThread(Camera &cam) {
  while (true) {
    const auto [fr, event] = cam.queue.pop();
    if (!fr) break;

    capnp::FlatArrayMessageReader reader(event->data);
    auto evt = reader.getRoot<cereal::Event>();
    auto eidx = capnp::AnyStruct::Reader(evt).getPointerSection()[0].getAs<cereal::EncodeIndex>();

    int segment_id = eidx.getSegmentId();
    uint32_t frame_id = eidx.getFrameId();
    if (auto yuv = getFrame(cam, fr, segment_id, frame_id)) {
      VisionIpcBufExtra extra = {
          .frame_id = frame_id,
          .timestamp_sof = eidx.getTimestampSof(),
          .timestamp_eof = eidx.getTimestampEof(),
      };
      vipc_server_->send(yuv, &extra);
    } else {
      rError("camera[%d] failed to get frame: %lu", cam.type, segment_id);
    }

    // Prefetch the next frame
    getFrame(cam, fr, segment_id + 1, frame_id + 1);

    --publishing_;
  }
}

VisionBuf *CameraServer::getFrame(Camera &cam, FrameReader *fr, int32_t segment_id, uint32_t frame_id) {
  // Check if the frame is cached
  auto buf_it = std::find_if(cam.cached_buf.begin(), cam.cached_buf.end(),
                             [frame_id](VisionBuf *buf) { return buf->get_frame_id() == frame_id; });
  if (buf_it != cam.cached_buf.end()) return *buf_it;

  VisionBuf *yuv_buf = vipc_server_->get_buffer(cam.stream_type);
  if (fr->get(segment_id, yuv_buf)) {
    yuv_buf->set_frame_id(frame_id);
    cam.cached_buf.insert(yuv_buf);
    return yuv_buf;
  }
  return nullptr;
}

void CameraServer::pushFrame(CameraType type, FrameReader *fr, const Event *event) {
  auto &cam = cameras_[type];
  if (cam.width != fr->width || cam.height != fr->height) {
    cam.width = fr->width;
    cam.height = fr->height;
    waitForSent();
    startVipcServer();
  }

  ++publishing_;
  cam.queue.push({fr, event});
}

void CameraServer::waitForSent() {
  while (publishing_ > 0) {
    std::this_thread::yield();
  }
}
