#pragma once

#include <memory>

#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLWidget>
#include <QThread>

#include "cereal/visionipc/visionipc_client.h"
#include "selfdrive/common/mat.h"
#include "selfdrive/common/visionimg.h"
#include "selfdrive/ui/ui.h"

class VIPCThread : public QThread {
  Q_OBJECT

public:
  VIPCThread(QObject *parent = nullptr) : stream_type_(VISION_STREAM_RGB_BACK), QThread(parent) {}

public slots:
  void receive(VisionStreamType stream_type);
  void disconnect();

signals:
  void connected(VisionIpcClient *);
  void recvd(VisionBuf *);

 protected:
  VisionStreamType stream_type_;
  std::unique_ptr<VisionIpcClient> vipc_client_;
};

class CameraViewWidget : public QOpenGLWidget, protected QOpenGLFunctions {
  Q_OBJECT

public:
  using QOpenGLWidget::QOpenGLWidget;
  explicit CameraViewWidget(VisionStreamType stream_type, bool zoom, QWidget* parent = nullptr);
  ~CameraViewWidget();
  void setStreamType(VisionStreamType type);

signals:
  void clicked();
  void frameUpdated();
  void updateFrame(VisionStreamType);
  void disconnectVipc();

protected:
  void paintGL() override;
  void resizeGL(int w, int h) override;
  void initializeGL() override;
  void hideEvent(QHideEvent *event) override;
  void mouseReleaseEvent(QMouseEvent *event) override;
  void updateFrameMat(int w, int h);
  bool vipc_connected = false;
  VisionStreamType stream_type;

protected slots:
  void vipcConnected(VisionIpcClient * vipc_client);
  void vipcRecvd(VisionBuf *buf);

private:
  bool zoomed_view;
  GLuint frame_vao, frame_vbo, frame_ibo;
  mat4 frame_mat;
  std::unique_ptr<EGLImageTexture> texture[UI_BUF_COUNT];
  QOpenGLShaderProgram *program;

  int stream_width = 0, stream_height = 0;
  VisionBuf *latest_frame = nullptr;
  VIPCThread *vipc_thread = nullptr;
};
