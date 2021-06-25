#include "selfdrive/ui/replay/camera.h"

#include <QDebug>

#include "selfdrive/camerad/cameras/camera_common.h"
#include "selfdrive/ui/replay/replay.h"

static const VisionStreamType stream_types[] = {
    [RoadCam] = VISION_STREAM_RGB_BACK,
    [DriverCam] = VISION_STREAM_RGB_FRONT,
    [WideRoadCam] = VISION_STREAM_RGB_WIDE,
};

class CameraServer::CameraState {
public:
  CameraState(VisionIpcServer *server, CameraType type, int w, int h) : width(w), height(h) {
    server->create_buffers(stream_types[type], UI_BUF_COUNT, true, width, height);
    thread = std::thread(&CameraState::run, this, server, type);
  }

  ~CameraState() {
    queue.push({nullptr, 0});
    thread.join();
  }

  void run(VisionIpcServer *server, CameraType type) {
    while (true) {
      const auto &[fr, encodeId] = queue.pop();
      if (!fr) break;

      if (uint8_t *dat = fr->get(encodeId)) {
        VisionIpcBufExtra extra = {};
        VisionBuf *buf = server->get_buffer(stream_types[type]);
        memcpy(buf->addr, dat, fr->getRGBSize());
        server->send(buf, &extra, false);
      } else {
        qDebug() << "failed get frame " << encodeId;
      }
    }
  }

  int width, height;
  std::thread thread;
  SafeQueue<std::pair<FrameReader *, uint32_t>> queue;

  friend inline bool frameChanged(const CameraState *c, const FrameReader *f) {
    return (!c && f) || (c && !f) || (c && f && (c->width != f->width || c->height != f->height));
  }
};

CameraServer::CameraServer() {
  device_id_ = cl_get_device_id(CL_DEVICE_TYPE_DEFAULT);
  context_ = CL_CHECK_ERR(clCreateContext(NULL, 1, &device_id_, NULL, NULL, &err));
}

CameraServer::~CameraServer() {
  stop();
  CL_CHECK(clReleaseContext(context_));
}

void CameraServer::pushFrame(CameraType type, const Segment *seg, uint32_t encodeFrameId) {
  if (seg_num != seg->seg_num) {
    // restart vipc server if frame changed. such as switched between qcameras and cameras.
    for (auto cam_type : ALL_CAMERAS) {
      if (frameChanged(camera_states_[cam_type], seg->frames[cam_type])) {
        stop();
        break;
      }
    }
    seg_num = seg->seg_num;
  }
  if (!vipc_server_) {
    vipc_server_ = new VisionIpcServer("camerad", device_id_, context_);
    for (auto cam_type : ALL_CAMERAS) {
      if (auto f = seg->frames[cam_type]) {
        camera_states_[cam_type] = new CameraState(vipc_server_, cam_type, f->width, f->height);
      }
    }
    vipc_server_->start_listener();
  }

  camera_states_[type]->queue.push({seg->frames[type], encodeFrameId});
}

void CameraServer::stop() {
  if (vipc_server_) {
    for (auto &cs : camera_states_) {
      delete cs;
      cs = nullptr;
    }
    delete vipc_server_;
    vipc_server_ = nullptr;
  }
}
