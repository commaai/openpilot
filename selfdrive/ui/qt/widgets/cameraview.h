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
const uint32_t INVALID_FRAME_ID = -1;
static_assert(FRAME_BUFFER_SIZE <= YUV_BUFFER_COUNT);

class CameraWidget : public QOpenGLWidget, protected QOpenGLFunctions {
  Q_OBJECT

public:
  explicit CameraWidget(std::string stream_name, VisionStreamType stream_type, bool zoom, QWidget* parent = nullptr);
  ~CameraWidget();
  void setBackgroundColor(const QColor &color) { bg = color; }
  void setFrameId(int frame_id) { draw_frame_id = frame_id; }
  void setStreamType(VisionStreamType type) { stream_type = type; }
  inline VisionStreamType streamType() const { return stream_type; }
  inline const std::set<VisionStreamType> &availableStreams() const { return available_streams; }
  VisionBuf *receiveFrame(uint64_t request_frame_id = INVALID_FRAME_ID);

signals:
  void vipcAvailableStreamsUpdated();
  void clicked();

protected:
  bool ensureConnection();
  void paintGL() override;
  void initializeGL() override;
  void vipcConnected();
  void clearFrames();
  void clearEGLImages();

  void resizeGL(int w, int h) override { updateFrameMat(); }
  void mouseReleaseEvent(QMouseEvent *event) override { emit clicked(); }
  virtual void updateFrameMat();
  void updateCalibration(const mat3 &calib);
  int glWidth() const { return width() * devicePixelRatio(); }
  int glHeight() const { return height() * devicePixelRatio(); }

  bool zoomed_view;
  GLuint frame_vao, frame_vbo, frame_ibo;
  GLuint textures[2];
  mat4 frame_mat = {};
  std::unique_ptr<QOpenGLShaderProgram> program;
  QColor bg = QColor("#000000");

#ifdef QCOM2
  std::map<int, EGLImageKHR> egl_images;
#endif

  std::string stream_name;
  int stream_width = 0;
  int stream_height = 0;
  int stream_stride = 0;
  VisionStreamType stream_type;

  // Calibration
  float x_offset = 0;
  float y_offset = 0;
  float zoom = 1.0;
  mat3 calibration = DEFAULT_CALIBRATION;
  mat3 intrinsic_matrix = FCAM_INTRINSIC_MATRIX;

  std::set<VisionStreamType> available_streams;
  std::unique_ptr<VisionIpcClient> vipc_client;
  std::deque<VisionBuf*> recent_frames;
  uint32_t draw_frame_id = INVALID_FRAME_ID;
  uint32_t prev_frame_id = INVALID_FRAME_ID;
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
