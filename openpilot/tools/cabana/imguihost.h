#pragma once

#include <chrono>
#include <cstdint>
#include <vector>

#include <QImage>
#include <QWidget>

#ifdef __APPLE__
struct GLFWwindow;
#else
#include <EGL/egl.h>
#endif

struct ImGuiContext;

// Hosts an imgui context inside the Qt shell: renders offscreen (EGL pbuffer on
// Linux, hidden GLFW window on Darwin), blits into the widget with QPainter.
// Qt is the platform layer; input is forwarded to ImGuiIO by the event handlers.
class ImGuiHost : public QWidget {
public:
  explicit ImGuiHost(QWidget *parent = nullptr);
  ~ImGuiHost() override;

protected:
  virtual void drawFrame() = 0;  // imgui commands, GL context current
  bool makeCurrent();
  void paintEvent(QPaintEvent *event) override;
  void mousePressEvent(QMouseEvent *event) override;
  void mouseReleaseEvent(QMouseEvent *event) override;
  void mouseMoveEvent(QMouseEvent *event) override;
  void wheelEvent(QWheelEvent *event) override;
  void leaveEvent(QEvent *event) override;

private:
  bool ensureContext(int w, int h);
  bool ensureSurface(int w, int h);
  void destroyGL();
  void forwardMouseButton(QMouseEvent *event, bool down);

#ifdef __APPLE__
  GLFWwindow *window = nullptr;
#else
  EGLDisplay display = EGL_NO_DISPLAY;
  EGLConfig config = nullptr;
  EGLContext context = EGL_NO_CONTEXT;
  EGLSurface surface = EGL_NO_SURFACE;
#endif
  ImGuiContext *imgui = nullptr;
  bool init_failed = false;
  bool backend_init = false;
  int fb_width = 0, fb_height = 0;
  QImage frame;
  std::vector<uint8_t> readback;
  std::chrono::steady_clock::time_point last_frame;
};
