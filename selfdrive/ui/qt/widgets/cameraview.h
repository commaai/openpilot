#pragma once

#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>

#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLWidget>
#include <QTimer>

#ifdef QCOM2
#define EGL_EGLEXT_PROTOTYPES
#define EGL_NO_X11
#define GL_TEXTURE_EXTERNAL_OES 0x8D65
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <drm/drm_fourcc.h>
#endif

#include "cereal/visionipc/visionipc_client.h"
#include "selfdrive/ui/ui.h"

class CameraWidget : public QOpenGLWidget, protected QOpenGLFunctions {
  Q_OBJECT

public:
  explicit CameraWidget(std::string stream_name, VisionStreamType stream_type, bool zoom, QWidget* parent = nullptr);
  ~CameraWidget();
  // Auto update based on timer, the default is disabled (i.e. false).
  void setAutoUpdate(bool enable);
  void setBackgroundColor(const QColor &color) { bg = color; }
  void setStreamType(VisionStreamType type) { requested_stream_type = type; }
  inline VisionStreamType streamType() const { return requested_stream_type; }
  inline const std::set<VisionStreamType> &availableStreams() const { return available_streams; }

signals:
  void vipcAvailableStreamsUpdated();

protected:
  bool receiveFrame(uint64_t  preferred_frame_id = 0);
  void paintGL() override;
  void initializeGL() override;
  void resizeGL(int w, int h) override { updateFrameMat(); }
  virtual void updateFrameMat();
  void updateCalibration(const mat3 &calib);
  void vipcConnected();

  int glWidth();
  int glHeight();

  bool zoomed_view;
  GLuint frame_vao, frame_vbo, frame_ibo;
  GLuint textures[2];
  mat4 frame_mat = {};
  std::unique_ptr<QOpenGLShaderProgram> program;
  QColor bg = QColor("#000000");

#ifdef QCOM2
  std::map<int, EGLImageKHR> egl_images;
#endif

  // vipc
  std::string stream_name;
  int stream_width = 0;
  int stream_height = 0;
  int stream_stride = 0;
  VisionStreamType requested_stream_type;
  std::set<VisionStreamType> available_streams;
  QTimer *vipc_timer = nullptr;
  bool conflate = false;
  std::unique_ptr<VisionIpcClient> vipc_client;
  VisionBuf *vipc_buffer_ = nullptr;
  uint64_t prev_frame_id = 0;

  // Calibration
  float x_offset = 0;
  float y_offset = 0;
  float zoom = 1.0;
  mat3 calibration = DEFAULT_CALIBRATION;
  mat3 intrinsic_matrix = FCAM_INTRINSIC_MATRIX;
};
