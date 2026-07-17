#include "tools/cabana/cameraview.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>

#include <QApplication>
#include <QPainter>

#include "common/yuv.h"

CameraWidget::CameraWidget(std::string stream_name, VisionStreamType type, QWidget* parent) :
                          stream_name(stream_name), active_stream_type(type), requested_stream_type(type), QWidget(parent) {
  setAttribute(Qt::WA_OpaquePaintEvent);
  qRegisterMetaType<std::set<VisionStreamType>>("availableStreams");
  QObject::connect(this, &CameraWidget::vipcThreadFrameReceived, this, &CameraWidget::vipcFrameReceived, Qt::QueuedConnection);
  QObject::connect(this, &CameraWidget::vipcAvailableStreamsUpdated, this, &CameraWidget::availableStreamsUpdated, Qt::QueuedConnection);
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

void CameraWidget::availableStreamsUpdated(std::set<VisionStreamType> streams) {
  available_streams = streams;
}

void CameraWidget::paintEvent(QPaintEvent *event) {
  QPainter p(this);
  p.fillRect(rect(), bg);

  std::lock_guard lk(frame_lock);
  if (!current_frame_ || stream_width <= 0 || stream_height <= 0) return;

  // Scale for aspect ratio
  float widget_ratio = (float)width() / height();
  float frame_ratio = (float)stream_width / stream_height;
  int w = std::lround(width() * std::min(frame_ratio / widget_ratio, 1.0f));
  int h = std::lround(height() * std::min(widget_ratio / frame_ratio, 1.0f));
  QRect video_rect((width() - w) / 2, (height() - h) / 2, w, h);

  // YUV (NV12) -> RGBA at stream resolution; the painter scales it into the video rect
  if (rgb_frame.width() != stream_width || rgb_frame.height() != stream_height) {
    rgb_frame = QImage(stream_width, stream_height, QImage::Format_RGBA8888);
  }
  yuv::nv12_to_rgba(current_frame_->y, stream_stride, current_frame_->uv, stream_stride,
                    rgb_frame.bits(), rgb_frame.bytesPerLine(), stream_width, stream_height);

  p.setRenderHint(QPainter::SmoothPixmapTransform);
  if (active_stream_type == VISION_STREAM_DRIVER) {
    // mirror driver camera horizontally
    const qreal cx = video_rect.x() + video_rect.width() / 2.0;
    p.translate(cx, 0);
    p.scale(-1, 1);
    p.translate(-cx, 0);
  }
  p.drawImage(video_rect, rgb_frame);
}

void CameraWidget::vipcFrameReceived() {
  update();
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
      emit vipcAvailableStreamsUpdated(streams);

      if (!vipc_client->connect(false)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        continue;
      }
      std::lock_guard lk(frame_lock);
      stream_width = vipc_client->buffers[0].width;
      stream_height = vipc_client->buffers[0].height;
      stream_stride = vipc_client->buffers[0].stride;
    }

    if (VisionBuf *buf = vipc_client->recv(&frame_meta, 100)) {
      {
        std::lock_guard lk(frame_lock);
        current_frame_ = buf;
        frame_meta_ = frame_meta;
      }
      emit vipcThreadFrameReceived();
    }
  }
}

void CameraWidget::clearFrames() {
  std::lock_guard lk(frame_lock);
  current_frame_ = nullptr;
  available_streams.clear();
}
