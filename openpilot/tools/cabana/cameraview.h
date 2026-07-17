#pragma once

#include <atomic>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <utility>

#include <QImage>
#include <QWidget>

#include "msgq/visionipc/visionipc_client.h"

class CameraWidget : public QWidget {
  Q_OBJECT

public:
  explicit CameraWidget(std::string stream_name, VisionStreamType stream_type, QWidget* parent = nullptr);
  ~CameraWidget();
  void setStreamType(VisionStreamType type) { requested_stream_type = type; }
  VisionStreamType getStreamType() { return active_stream_type; }
  void stopVipcThread();

signals:
  void clicked();
  void vipcThreadFrameReceived();
  void vipcAvailableStreamsUpdated(std::set<VisionStreamType>);

protected:
  void paintEvent(QPaintEvent *event) override;
  void showEvent(QShowEvent *event) override;
  void mouseReleaseEvent(QMouseEvent *event) override { emit clicked(); }
  void vipcThread();
  void clearFrames();

  QColor bg = Qt::black;
  QImage rgb_frame;

  std::string stream_name;
  int stream_width = 0;
  int stream_height = 0;
  int stream_stride = 0;
  std::atomic<VisionStreamType> active_stream_type;
  std::atomic<VisionStreamType> requested_stream_type;
  std::set<VisionStreamType> available_streams;
  std::thread vipc_thread;
  std::atomic<bool> vipc_exit = false;
  std::recursive_mutex frame_lock;
  VisionBuf* current_frame_ = nullptr;
  VisionIpcBufExtra frame_meta_ = {};

protected slots:
  void vipcFrameReceived();
  void availableStreamsUpdated(std::set<VisionStreamType> streams);
};

Q_DECLARE_METATYPE(std::set<VisionStreamType>);
