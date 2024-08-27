#pragma once

#include <memory>
#include <set>
#include <string>

#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLWidget>
#include "common/mat.h"
#include "msgq/visionipc/visionipc_client.h"

class CameraWidget : public QOpenGLWidget, protected QOpenGLFunctions {
  Q_OBJECT

public:
  using QOpenGLWidget::QOpenGLWidget;
  explicit CameraWidget(const std::string &stream_name, VisionStreamType stream_type, QWidget* parent = nullptr);
  void setStreamType(VisionStreamType type);
  VisionStreamType getStreamType() const { return vipc_client->type; }
  void recvFrame();

signals:
  void clicked();
  void vipcAvailableStreamsUpdated(std::set<VisionStreamType> streams);

protected:
  void initializeGL() override;
  void paintGL() override;
  void mouseReleaseEvent(QMouseEvent *event) override { emit clicked(); }
  void updateFrameMat();
  inline int glWidth() const { return width() * devicePixelRatio(); };
  inline int glHeight() const { return height() * devicePixelRatio(); }

  GLuint frame_vao, frame_vbo, frame_ibo;
  GLuint textures[2];
  mat4 frame_mat = {};
  std::unique_ptr<QOpenGLShaderProgram> program;

  std::string stream_name;
  int stream_width = 0;
  int stream_height = 0;
  int stream_stride = 0;
  std::unique_ptr<VisionIpcClient> vipc_client;
  VisionBuf *frame = nullptr;
};

Q_DECLARE_METATYPE(std::set<VisionStreamType>);
