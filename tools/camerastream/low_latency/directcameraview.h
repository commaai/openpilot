#pragma once

#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLWidget>
#include <QThread>

#include <cuda.h>
#include <cuviddec.h>
#include <nvcuvid.h>
#include "simple_decoder.h"


class DirectCameraViewWidget : public QOpenGLWidget, protected QOpenGLFunctions {
  Q_OBJECT
public:
  using QOpenGLWidget::QOpenGLWidget;
  DirectCameraViewWidget(const char* stream_name, const char* addr);

signals:
  void vipcThreadFrameReceived(unsigned char *dat, int len);

protected:
  void paintGL() override;
  void initializeGL() override;
  void showEvent(QShowEvent *event) override;

  const char *stream_name, *addr;
  std::unique_ptr<SimpleDecoder> decoder;

  void vipcThread();
  QThread *vipc_thread = nullptr;

  GLuint frame_vao, frame_vbo, frame_ibo;
  std::unique_ptr<QOpenGLShaderProgram> program;
  GLuint textures[3];

  CUgraphicsResource res[3];
  CUdeviceptr dpSrcFrame;
  bool is_new;
  uint64_t st;

  QColor bg = QColor("#000000");

protected slots:
  void vipcFrameReceived(unsigned char *dat, int len);
};