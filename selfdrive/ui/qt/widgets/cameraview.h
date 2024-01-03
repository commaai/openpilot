#pragma once

#include <deque>
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
#include "system/camerad/cameras/camera_common.h"
#include "selfdrive/ui/ui.h"

const int FRAME_BUFFER_SIZE = 5;
static_assert(FRAME_BUFFER_SIZE <= YUV_BUFFER_COUNT);

class CameraWidget : public QOpenGLWidget, protected QOpenGLFunctions {
  Q_OBJECT

public:
  explicit CameraWidget(std::string stream_name, VisionStreamType stream_type, bool zoom, QWidget* parent = nullptr);
  ~CameraWidget();
  void disconnectVipc();
  void setBackgroundColor(const QColor &color) { bg = color; }
  void setStreamType(VisionStreamType type) { requested_stream_type = type; }
  inline VisionStreamType streamType() const { return requested_stream_type; }
  inline const std::set<VisionStreamType> &availableStreams() const { return available_streams; }
  bool receiveFrame(uint64_t request_frame_id = 0);

signals:
  void vipcAvailableStreamsUpdated();
  void clicked();

protected:
  void paintGL() override;
  void initializeGL() override;
  void resizeGL(int w, int h) override { updateFrameMat(); }
  void mouseReleaseEvent(QMouseEvent *event) override { emit clicked(); }
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
  std::unique_ptr<VisionIpcClient> vipc_client;
  std::deque<std::pair<uint32_t, VisionBuf*>> recent_frames;
  VisionBuf *frame = nullptr;
  uint64_t frame_id = 0;
  uint64_t prev_frame_id = 0;

  // Calibration
  float x_offset = 0;
  float y_offset = 0;
  float zoom = 1.0;
  mat3 calibration = DEFAULT_CALIBRATION;
  mat3 intrinsic_matrix = FCAM_INTRINSIC_MATRIX;
};

// update frames based on timer
class CameraView : public CameraWidget {
  Q_OBJECT
public:
  CameraView(const std::string &name, VisionStreamType stream_type, bool zoom, QWidget *parent = nullptr);
  void showEvent(QShowEvent *event) override;
  void hideEvent(QHideEvent *event) override;

private:
  QTimer *timer;
};
