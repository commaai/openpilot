#pragma once

#include <atomic>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <utility>

#include <QOpenGLFunctions>
#include <QOpenGLWidget>

#include "msgq/visionipc/visionipc_client.h"

class CameraWidget : public QOpenGLWidget, protected QOpenGLFunctions {
  Q_OBJECT

public:
  using QOpenGLWidget::QOpenGLWidget;
  explicit CameraWidget(std::string stream_name, VisionStreamType stream_type, QWidget* parent = nullptr);
  ~CameraWidget();
  void setStreamType(VisionStreamType type) { requested_stream_type = type; }
  VisionStreamType getStreamType() { return active_stream_type; }
  void stopVipcThread();

signals:
  void clicked();
  void vipcThreadConnected(VisionIpcClient *);
  void vipcThreadFrameReceived();
  void vipcAvailableStreamsUpdated(std::set<VisionStreamType>);

protected:
  void paintGL() override;
  void initializeGL() override;
  void showEvent(QShowEvent *event) override;
  void mouseReleaseEvent(QMouseEvent *event) override { emit clicked(); }
  void vipcThread();
  void clearFrames();

  GLuint frame_vao = 0, frame_vbo = 0, frame_ibo = 0;
  GLuint textures[2] = {};
  GLuint shader_program = 0;
  GLint transform_uniform = -1;
  QColor bg = Qt::black;

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
  void vipcConnected(VisionIpcClient *vipc_client);
  void vipcFrameReceived();
  void availableStreamsUpdated(std::set<VisionStreamType> streams);
};

Q_DECLARE_METATYPE(std::set<VisionStreamType>);
