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
  void hideEvent(QHideEvent *event) override { stopVipcThread(); }
  void mouseReleaseEvent(QMouseEvent *event) override { emit clicked(); }
  void vipcThread();
  void clearFrames();

  QColor bg = Qt::black;
  QImage rgb_frame;   // written by vipc thread, drawn by GUI thread; guarded by frame_lock
  QImage rgb_back;    // vipc thread only

  std::string stream_name;
  std::atomic<VisionStreamType> active_stream_type;
  std::atomic<VisionStreamType> requested_stream_type;
  std::set<VisionStreamType> available_streams;
  std::thread vipc_thread;
  std::atomic<bool> vipc_exit = false;
  std::mutex frame_lock;

protected slots:
  void vipcFrameReceived();
  void availableStreamsUpdated(std::set<VisionStreamType> streams);
};

Q_DECLARE_METATYPE(std::set<VisionStreamType>);
