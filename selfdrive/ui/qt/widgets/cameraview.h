#pragma once

#include <memory>
#include <mutex>

#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLWidget>
#include <QThread>

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
  using QOpenGLWidget::QOpenGLWidget;
  explicit CameraWidget(std::string stream_name, VisionStreamType stream_type, bool zoom, QWidget* parent = nullptr);
  ~CameraWidget();
  void setBackgroundColor(const QColor &color) { bg = color; }
  void setFrameId(int frame_id) { draw_frame_id = frame_id; }
  void setStreamType(VisionStreamType type) { requested_stream_type = type; }
  VisionStreamType getStreamType() { return active_stream_type; }
  void stopVipcThread();

signals:
  void clicked();
  void vipcThreadConnected(VisionIpcClient *);
  void vipcThreadFrameReceived();
  void vipcAvailableStreamsUpdated(std::set<VisionStreamType>);

protected:
  void paintGL() override;
  void initializeGL() override;
  void resizeGL(int w, int h) override { updateFrameMat(); }
  void showEvent(QShowEvent *event) override;
  void mouseReleaseEvent(QMouseEvent *event) override { emit clicked(); }
  virtual void updateFrameMat();
  void updateCalibration(const mat3 &calib);
  void vipcThread();
  void clearFrames();

  int glWidth();
  int glHeight();

  bool zoomed_view;
  GLuint frame_vao, frame_vbo, frame_ibo;
  GLuint textures[2];
  mat4 frame_mat;
  std::unique_ptr<QOpenGLShaderProgram> program;
  QColor bg = QColor("#000000");

#ifdef QCOM2
  EGLDisplay egl_display;
  std::map<int, EGLImageKHR> egl_images;
#endif

  std::string stream_name;
  int stream_width = 0;
  int stream_height = 0;
  int stream_stride = 0;
  std::atomic<VisionStreamType> active_stream_type;
  std::atomic<VisionStreamType> requested_stream_type;
  std::set<VisionStreamType> available_streams;
  QThread *vipc_thread = nullptr;

  float frames_wide = 0;
  bool ready_to_switch = false;

  // Calibration
  float x_offset = 0;
  float y_offset = 0;
  float zoom = 1.0;
  mat3 calibration = DEFAULT_CALIBRATION;
  mat3 intrinsic_matrix = fcam_intrinsic_matrix;

  std::recursive_mutex frame_lock;
  std::deque<std::pair<uint32_t, VisionBuf*>> frames;
  uint32_t draw_frame_id = 0;
  uint32_t prev_frame_id = 0;

protected slots:
  void vipcConnected(VisionIpcClient *vipc_client);
  void vipcFrameReceived();
  void availableStreamsUpdated(std::set<VisionStreamType> streams);
};

Q_DECLARE_METATYPE(std::set<VisionStreamType>);
