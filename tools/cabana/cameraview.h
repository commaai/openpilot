#pragma once

#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <utility>

#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLWidget>
#include <QThread>

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

  GLuint frame_vao, frame_vbo, frame_ibo;
  GLuint textures[2];
  std::unique_ptr<QOpenGLShaderProgram> shader_program_;
  QColor bg = Qt::black;

  std::string stream_name;
  int stream_width = 0;
  int stream_height = 0;
  int stream_stride = 0;
  std::atomic<VisionStreamType> active_stream_type;
  std::atomic<VisionStreamType> requested_stream_type;
  std::set<VisionStreamType> available_streams;
  QThread *vipc_thread = nullptr;
  std::recursive_mutex frame_lock;
  VisionBuf* current_frame_ = nullptr;
  VisionIpcBufExtra frame_meta_ = {};

protected slots:
  void vipcConnected(VisionIpcClient *vipc_client);
  void vipcFrameReceived();
  void availableStreamsUpdated(std::set<VisionStreamType> streams);
};

Q_DECLARE_METATYPE(std::set<VisionStreamType>);
