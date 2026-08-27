#include "tools/cabana/cameraview.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>

#include <QApplication>
#include <QPainter>

#include "common/yuv.h"
#include "tools/cabana/utils/util.h"

CameraWidget::CameraWidget(std::string stream_name, VisionStreamType type, QWidget* parent) :
                          stream_name(stream_name), active_stream_type(type), requested_stream_type(type), QWidget(parent) {
  setAttribute(Qt::WA_OpaquePaintEvent);
  QObject::connect(QApplication::instance(), &QCoreApplication::aboutToQuit, this, &CameraWidget::stopVipcThread);
}

CameraWidget::~CameraWidget() {
  stopVipcThread();
}

void CameraWidget::showEvent(QShowEvent *event) {
  if (!vipc_thread.joinable()) {
    clearFrames();
    vipc_exit = false;
    vipc_thread = std::thread(&CameraWidget::vipcThread, this);
  }
}

void CameraWidget::stopVipcThread() {
  vipc_exit = true;
  if (vipc_thread.joinable()) {
    vipc_thread.join();
  }
}

void CameraWidget::paintEvent(QPaintEvent *event) {
  QPainter p(this);
  p.fillRect(rect(), bg);

  std::lock_guard lk(frame_lock);
  if (rgb_frame.isNull()) return;

  // Scale for aspect ratio
  float widget_ratio = (float)width() / height();
  float frame_ratio = (float)rgb_frame.width() / rgb_frame.height();
  int w = std::lround(width() * std::min(frame_ratio / widget_ratio, 1.0f));
  int h = std::lround(height() * std::min(widget_ratio / frame_ratio, 1.0f));
  QRect video_rect((width() - w) / 2, (height() - h) / 2, w, h);

  p.setRenderHint(QPainter::SmoothPixmapTransform);
  if (active_stream_type == VISION_STREAM_CABIN) {
    // mirror cabin camera horizontally
    const qreal cx = video_rect.x() + video_rect.width() / 2.0;
    p.translate(cx, 0);
    p.scale(-1, 1);
    p.translate(-cx, 0);
  }
  p.drawImage(video_rect, rgb_frame);
}

void CameraWidget::vipcThread() {
  VisionStreamType cur_stream = requested_stream_type;
  std::unique_ptr<VisionIpcClient> vipc_client;
  VisionIpcBufExtra frame_meta = {};

  while (!vipc_exit) {
    if (!vipc_client || cur_stream != requested_stream_type) {
      clearFrames();
      fprintf(stderr, "connecting to stream %d, was connected to %d\n",
              (int)requested_stream_type, (int)cur_stream);
      cur_stream = requested_stream_type;
      vipc_client.reset(new VisionIpcClient(stream_name, cur_stream, false));
    }
    active_stream_type = cur_stream;

    if (!vipc_client->connected) {
      clearFrames();
      auto streams = VisionIpcClient::getAvailableStreams(stream_name, false);
      if (streams.empty()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        continue;
      }
      utils::runOnMainThread([this, alive = std::weak_ptr<bool>(alive_), streams]() {
        if (alive.expired()) return;
        available_streams = streams;
        availableStreamsUpdated(streams);
      });

      if (!vipc_client->connect(false)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        continue;
      }
    }

    if (VisionBuf *buf = vipc_client->recv(&frame_meta, 100)) {
      // NV12 -> RGBA once per frame on the receive thread; paint just draws the image
      if (rgb_back.width() != (int)buf->width || rgb_back.height() != (int)buf->height) {
        rgb_back = QImage(buf->width, buf->height, QImage::Format_RGBA8888);
      }
      yuv::nv12_to_rgba(buf->y, buf->stride, buf->uv, buf->stride,
                        rgb_back.bits(), rgb_back.bytesPerLine(), buf->width, buf->height);
      {
        std::lock_guard lk(frame_lock);
        rgb_frame.swap(rgb_back);
      }
      utils::runOnMainThread([this, alive = std::weak_ptr<bool>(alive_)]() {
        if (!alive.expired()) update();
      });
    }
  }
}

void CameraWidget::clearFrames() {
  std::lock_guard lk(frame_lock);
  rgb_frame = QImage();
  rgb_back = QImage();
  available_streams.clear();
}
