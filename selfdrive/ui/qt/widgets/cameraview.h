#pragma once

#include <memory>
#include <mutex>
#include <condition_variable>

#include <QOpenGLFunctions>
#include <QOpenGLWidget>
#include <QThread>

#include "cereal/visionipc/visionipc_client.h"
#include "selfdrive/common/glutil.h"
#include "selfdrive/common/mat.h"
#include "selfdrive/common/visionimg.h"
#include "selfdrive/ui/ui.h"

class Render;
class CameraViewWidget : public QOpenGLWidget {
  Q_OBJECT
public:
  explicit CameraViewWidget(VisionStreamType stream_type, QWidget* parent = nullptr);
  ~CameraViewWidget();

signals:
 void frameUpdated();
 void renderRequested(bool hidden);

public slots:
  void moveContextToThread();

protected:
  void paintEvent(QPaintEvent* event) override {}
  void hideEvent(QHideEvent *event) override;
  QThread *thread_;
  Render *render_;
};

class Render : public QObject, protected QOpenGLFunctions {
  Q_OBJECT
public:
  Render(VisionStreamType stream_type, CameraViewWidget *w);
  ~Render();

signals:
  void contextWanted();

private:
  void render(bool hidden);
  void updateFrame();
  bool frameUpdated() const { return latest_frame != nullptr; };
  void initialize();
  void draw();
  bool inited_ = false;
  CameraViewWidget * glWindow_;
  std::mutex renderMutex_;
  std::condition_variable grabCond_;

  VisionBuf *latest_frame = nullptr;
  GLuint frame_vao, frame_vbo, frame_ibo;
  mat4 frame_mat;
  std::unique_ptr<VisionIpcClient> vipc_client;
  std::unique_ptr<EGLImageTexture> texture[UI_BUF_COUNT];
  std::unique_ptr<GLShader> gl_shader;

  VisionStreamType stream_type;
  bool exiting_ = false;
  friend class CameraViewWidget;
};
