#pragma once

#include <memory>

#include <QOpenGLFunctions>
#include <QOpenGLWidget>

#include "cereal/visionipc/visionipc_client.h"
#include "selfdrive/common/glutil.h"
#include "selfdrive/common/mat.h"
#include "selfdrive/common/visionimg.h"
#include "selfdrive/ui/ui.h"

class CameraViewWidget : public QOpenGLWidget, protected QOpenGLFunctions {
Q_OBJECT

public:
  using QOpenGLWidget::QOpenGLWidget;
  explicit CameraViewWidget(VisionStreamType stream_type, QWidget* parent = nullptr);
  ~CameraViewWidget();

signals:
 void frameUpdated();

protected:
  void paintGL() override;
  void initializeGL() override;
  void showEvent(QShowEvent *event) override;
  void hideEvent(QHideEvent *event) override;

protected slots:
  void updateFrame();

private:
  VisionBuf *latest_frame = nullptr;
  GLuint frame_vao, frame_vbo, frame_ibo;
  mat4 frame_mat;
  std::unique_ptr<VisionIpcClient> vipc_client;
  std::unique_ptr<EGLImageTexture> texture[UI_BUF_COUNT];
  std::unique_ptr<GLShader> gl_shader;

  VisionStreamType stream_type;
  QTimer* timer;
};
