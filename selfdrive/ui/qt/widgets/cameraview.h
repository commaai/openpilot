#pragma once

#include <memory>

#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLWidget>
#include <QThread>
#include "cereal/visionipc/visionipc_client.h"
#include "selfdrive/common/visionimg.h"
#include "selfdrive/ui/ui.h"

class CameraViewWidget : public QOpenGLWidget, protected QOpenGLFunctions {
  Q_OBJECT

public:
  using QOpenGLWidget::QOpenGLWidget;
  explicit CameraViewWidget(VisionStreamType stream_type, bool zoom, QWidget* parent = nullptr);
  ~CameraViewWidget();
  void setStreamType(VisionStreamType type) { stream_type = type; }
  void setBackgroundColor(const QColor &color) { bg = color; }

signals:
  void clicked();
  void frameUpdated();

protected:
  void paintGL() override;
  void initializeGL() override;
  void resizeGL(int w, int h) override { updateFrameMat(w, h); }
  void showEvent(QShowEvent *event) override;
  void hideEvent(QHideEvent *event) override;
  void mouseReleaseEvent(QMouseEvent *event) override { emit clicked(); }
  void updateFrameMat(int w, int h);
  void vipcThread();

  bool zoomed_view;
  VisionBuf *latest_frame = nullptr;
  GLuint frame_vao, frame_vbo, frame_ibo;
  mat4 frame_mat;
  std::unique_ptr<EGLImageTexture> texture[UI_BUF_COUNT];
  QOpenGLShaderProgram *program;
  QColor bg = QColor("#000000");

  int stream_width = 0, stream_height = 0;
  std::atomic<VisionStreamType> stream_type;
  std::unique_ptr<VisionIpcClient> vipc_client_;
  QThread *vipc_thread_ = nullptr;

protected slots:
  void vipcConnected(VisionIpcClient * vipc_client);
  void vipcFrameReceived(VisionBuf *buf);

Q_SIGNALS:
  void vipcThreadConnected(VisionIpcClient *);
  void vipcThreadFrameReceived(VisionBuf *);
};
