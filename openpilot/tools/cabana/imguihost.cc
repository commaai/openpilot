#include "tools/cabana/imguihost.h"

#include <algorithm>
#include <cfloat>
#include <cstring>

#include <QDebug>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QPainter>

#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_opengl3_loader.h"
#include "imgui_internal.h"

#ifdef __APPLE__
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

static int glfw_refs = 0;
#else
#include <EGL/eglext.h>

// EGL display is shared across hosts; terminate when the last one goes away.
static EGLDisplay egl_display = EGL_NO_DISPLAY;
static int egl_display_refs = 0;

static EGLDisplay acquireDisplay() {
  if (egl_display == EGL_NO_DISPLAY) {
    auto get_platform_display = (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
    EGLDisplay d = EGL_NO_DISPLAY;
    if (get_platform_display) d = get_platform_display(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, nullptr);
    if (d == EGL_NO_DISPLAY) d = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (d == EGL_NO_DISPLAY || !eglInitialize(d, nullptr, nullptr)) return EGL_NO_DISPLAY;
    egl_display = d;
  }
  ++egl_display_refs;
  return egl_display;
}

static void releaseDisplay() {
  if (--egl_display_refs == 0) {
    eglTerminate(egl_display);
    egl_display = EGL_NO_DISPLAY;
  }
}
#endif

ImGuiHost::ImGuiHost(QWidget *parent) : QWidget(parent) {
  setMouseTracking(true);
  setFocusPolicy(Qt::ClickFocus);
}

ImGuiHost::~ImGuiHost() {
  destroyGL();
}

bool ImGuiHost::makeCurrent() {
#ifdef __APPLE__
  if (!window) return false;
  glfwMakeContextCurrent(window);
#else
  if (context == EGL_NO_CONTEXT || !eglMakeCurrent(display, surface, surface, context)) return false;
#endif
  if (imgui) ImGui::SetCurrentContext(imgui);
  return true;
}

bool ImGuiHost::ensureSurface(int w, int h) {
#ifdef __APPLE__
  int fw = 0, fh = 0;
  glfwGetFramebufferSize(window, &fw, &fh);
  if (fw < w || fh < h) return false;  // hidden-window resize is async; created oversized instead
#else
  if (surface == EGL_NO_SURFACE || w != fb_width || h != fb_height) {
    if (surface != EGL_NO_SURFACE) {
      eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
      eglDestroySurface(display, surface);
    }
    const EGLint attribs[] = {EGL_WIDTH, w, EGL_HEIGHT, h, EGL_NONE};
    surface = eglCreatePbufferSurface(display, config, attribs);
    if (surface == EGL_NO_SURFACE) return false;
  }
#endif
  fb_width = w;
  fb_height = h;
  return true;
}

bool ImGuiHost::ensureContext(int w, int h) {
  if (imgui) return true;
  if (init_failed) return false;

  auto init = [&]() {
#ifdef __APPLE__
    if (!glfwInit()) return false;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    window = glfwCreateWindow(std::max(2048, w), std::max(2048, h), "cabana imgui", nullptr, nullptr);
    if (!window) return false;
    ++glfw_refs;
#else
    if ((display = acquireDisplay()) == EGL_NO_DISPLAY) return false;
    eglBindAPI(EGL_OPENGL_API);
    const EGLint config_attribs[] = {
      EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
      EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
      EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
      EGL_NONE,
    };
    EGLint num_configs = 0;
    if (!eglChooseConfig(display, config_attribs, &config, 1, &num_configs) || num_configs == 0) return false;
    const EGLint context_attribs[] = {
      EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, 3,
      EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
      EGL_NONE,
    };
    context = eglCreateContext(display, config, EGL_NO_CONTEXT, context_attribs);
    if (context == EGL_NO_CONTEXT) return false;
#endif
    if (!ensureSurface(w, h) || !makeCurrent()) return false;
    imgui = ImGui::CreateContext();
    ImGui::SetCurrentContext(imgui);
    ImGuiIO &io = ImGui::GetIO();
    io.IniFilename = io.LogFilename = nullptr;
    if (!ImGui_ImplOpenGL3_Init("#version 330")) return false;
    last_frame = std::chrono::steady_clock::now();
    return true;
  };
  if (!init()) {
    destroyGL();
    init_failed = true;
    qWarning() << "ImGuiHost: GL init failed";
    return false;
  }
  return true;
}

void ImGuiHost::destroyGL() {
  if (imgui) {
    if (makeCurrent()) ImGui_ImplOpenGL3_Shutdown();
    ImGui::DestroyContext(imgui);
    imgui = nullptr;
  }
#ifdef __APPLE__
  if (window) {
    glfwDestroyWindow(window);
    window = nullptr;
    if (--glfw_refs == 0) glfwTerminate();
  }
#else
  if (display != EGL_NO_DISPLAY) {
    eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    if (surface != EGL_NO_SURFACE) eglDestroySurface(display, surface);
    if (context != EGL_NO_CONTEXT) eglDestroyContext(display, context);
    surface = EGL_NO_SURFACE;
    context = EGL_NO_CONTEXT;
    releaseDisplay();
    display = EGL_NO_DISPLAY;
  }
#endif
}

void ImGuiHost::paintEvent(QPaintEvent *event) {
  const qreal dpr = devicePixelRatioF();
  const int w = std::max(1, (int)(width() * dpr));
  const int h = std::max(1, (int)(height() * dpr));
  if (!ensureContext(w, h) || !ensureSurface(w, h) || !makeCurrent()) {
    QPainter(this).fillRect(rect(), Qt::black);
    return;
  }

  ImGuiIO &io = ImGui::GetIO();
  io.DisplaySize = ImVec2(width(), height());
  io.DisplayFramebufferScale = ImVec2(dpr, dpr);
  auto now = std::chrono::steady_clock::now();
  io.DeltaTime = std::clamp(std::chrono::duration<float>(now - last_frame).count(), 1e-4f, 1.0f);
  last_frame = now;

  ImGui_ImplOpenGL3_NewFrame();
  ImGui::NewFrame();
  drawFrame();
  ImGui::Render();
  glViewport(0, 0, w, h);
  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT);
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

  // read back, flipping GL's bottom-left origin
  if (frame.size() != QSize(w, h)) {
    frame = QImage(w, h, QImage::Format_RGBA8888);
    readback.resize((size_t)w * h * 4);
  }
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, readback.data());
  const size_t stride = (size_t)w * 4;
  for (int y = 0; y < h; ++y) {
    memcpy(frame.scanLine(y), readback.data() + (size_t)(h - 1 - y) * stride, stride);
  }
  frame.setDevicePixelRatio(dpr);
  QPainter(this).drawImage(0, 0, frame);

  // imgui trickles queued input over multiple frames (e.g. press/release of a click)
  if (imgui->InputEventsQueue.Size > 0) update();
}

static ImGuiKey toImGuiKey(int key) {
  switch (key) {
    case Qt::Key_Tab: return ImGuiKey_Tab;
    case Qt::Key_Left: return ImGuiKey_LeftArrow;
    case Qt::Key_Right: return ImGuiKey_RightArrow;
    case Qt::Key_Up: return ImGuiKey_UpArrow;
    case Qt::Key_Down: return ImGuiKey_DownArrow;
    case Qt::Key_PageUp: return ImGuiKey_PageUp;
    case Qt::Key_PageDown: return ImGuiKey_PageDown;
    case Qt::Key_Home: return ImGuiKey_Home;
    case Qt::Key_End: return ImGuiKey_End;
    case Qt::Key_Delete: return ImGuiKey_Delete;
    case Qt::Key_Backspace: return ImGuiKey_Backspace;
    case Qt::Key_Space: return ImGuiKey_Space;
    case Qt::Key_Return: return ImGuiKey_Enter;
    case Qt::Key_Enter: return ImGuiKey_KeypadEnter;
    case Qt::Key_Escape: return ImGuiKey_Escape;
    default:
      if (key >= Qt::Key_A && key <= Qt::Key_Z) return (ImGuiKey)(ImGuiKey_A + (key - Qt::Key_A));
      if (key >= Qt::Key_0 && key <= Qt::Key_9) return (ImGuiKey)(ImGuiKey_0 + (key - Qt::Key_0));
      return ImGuiKey_None;
  }
}

void ImGuiHost::forwardMouseButton(QMouseEvent *event, bool down) {
  if (!imgui) return;
  int button = event->button() == Qt::LeftButton ? 0 : event->button() == Qt::RightButton ? 1
             : event->button() == Qt::MiddleButton ? 2 : -1;
  if (button < 0) return;
  ImGui::SetCurrentContext(imgui);
  ImGuiIO &io = ImGui::GetIO();
  io.AddMousePosEvent(event->x(), event->y());
  io.AddMouseButtonEvent(button, down);
}

void ImGuiHost::forwardKey(QKeyEvent *event, bool down) {
  if (!imgui) return;
  ImGui::SetCurrentContext(imgui);
  ImGuiIO &io = ImGui::GetIO();
  io.AddKeyEvent(ImGuiMod_Ctrl, event->modifiers() & Qt::ControlModifier);
  io.AddKeyEvent(ImGuiMod_Shift, event->modifiers() & Qt::ShiftModifier);
  io.AddKeyEvent(ImGuiMod_Alt, event->modifiers() & Qt::AltModifier);
  io.AddKeyEvent(ImGuiMod_Super, event->modifiers() & Qt::MetaModifier);
  if (ImGuiKey key = toImGuiKey(event->key()); key != ImGuiKey_None) {
    io.AddKeyEvent(key, down);
  }
  if (down && !(event->modifiers() & Qt::ControlModifier) && !event->text().isEmpty()) {
    io.AddInputCharactersUTF8(event->text().toUtf8().constData());
  }
}

void ImGuiHost::mousePressEvent(QMouseEvent *event) {
  forwardMouseButton(event, true);
  QWidget::mousePressEvent(event);
  update();
}

void ImGuiHost::mouseReleaseEvent(QMouseEvent *event) {
  forwardMouseButton(event, false);
  QWidget::mouseReleaseEvent(event);
  update();
}

void ImGuiHost::mouseMoveEvent(QMouseEvent *event) {
  if (imgui) {
    ImGui::SetCurrentContext(imgui);
    ImGui::GetIO().AddMousePosEvent(event->x(), event->y());
  }
  QWidget::mouseMoveEvent(event);
  update();
}

void ImGuiHost::wheelEvent(QWheelEvent *event) {
  if (imgui) {
    ImGui::SetCurrentContext(imgui);
    ImGui::GetIO().AddMouseWheelEvent(event->angleDelta().x() / 120.0f, event->angleDelta().y() / 120.0f);
  }
  QWidget::wheelEvent(event);
  update();
}

void ImGuiHost::keyPressEvent(QKeyEvent *event) {
  forwardKey(event, true);
  QWidget::keyPressEvent(event);
  update();
}

void ImGuiHost::keyReleaseEvent(QKeyEvent *event) {
  forwardKey(event, false);
  QWidget::keyReleaseEvent(event);
  update();
}

void ImGuiHost::leaveEvent(QEvent *event) {
  if (imgui) {
    ImGui::SetCurrentContext(imgui);
    ImGui::GetIO().AddMousePosEvent(-FLT_MAX, -FLT_MAX);
  }
  QWidget::leaveEvent(event);
  update();
}

void ImGuiHost::focusOutEvent(QFocusEvent *event) {
  if (imgui) {
    ImGui::SetCurrentContext(imgui);
    ImGui::GetIO().AddFocusEvent(false);
  }
  QWidget::focusOutEvent(event);
  update();
}
