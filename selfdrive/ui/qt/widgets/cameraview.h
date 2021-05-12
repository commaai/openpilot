#pragma once

#include <memory>

#include <QOpenGLFunctions>
#include <QOpenGLWidget>

#include "cereal/visionipc/visionipc_client.h"
#include "selfdrive/camerad/cameras/camera_common.h"
#include "selfdrive/common/glutil.h"
#include "selfdrive/common/mat.h"
#include "selfdrive/common/modeldata.h"
#include "selfdrive/ui/ui.h"

class CameraViewWidget : public QOpenGLWidget, protected QOpenGLFunctions {
  Q_OBJECT
 public:
  using QOpenGLWidget::QOpenGLWidget;
  CameraViewWidget(QWidget *parent);
  virtual ~CameraViewWidget();

 protected:
  void paintGL() override;
  void initializeGL() override;

  std::unique_ptr<VisionIpcClient> vipc_client;
  GLuint frame_vao, frame_vbo, frame_ibo;
  mat4 frame_mat;
  inline static std::unique_ptr<GLShader> gl_shader;

};

// container window for the NVG UI
class RoadView : public CameraViewWidget {
  Q_OBJECT

public:
  // using QOpenGLWidget::QOpenGLWidget;
  explicit RoadView(QWidget* parent = 0);

protected:
  void paintGL() override;
  void initializeGL() override;

private:
  double prev_draw_t = 0;

public slots:
  void update(const UIState &s);
};

class DriverView : public CameraViewWidget {
Q_OBJECT
public:
  DriverView(QWidget *parent) : CameraViewWidget(parent) {}
  void paintGL() override;
  // void initializeGL() override;
};
