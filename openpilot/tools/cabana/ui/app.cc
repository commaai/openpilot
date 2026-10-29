#include "tools/cabana/ui/app.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <thread>
#include <utility>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#ifdef __EMSCRIPTEN__
#include <GLES3/gl3.h>
#else
#include "imgui_impl_opengl3_loader.h"
#endif
#include "implot.h"
#include <GLFW/glfw3.h>
#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <GLFW/emscripten_glfw3.h>
#endif

#include "tools/cabana/settings.h"
#include "tools/cabana/ui/inistate.h"
#include "tools/cabana/ui/util.h"
#include "tools/cabana/ui/mainwin.h"
#include "tools/cabana/utils/util.h"

namespace {

std::atomic<bool> g_signal_exit{false};
std::vector<KeyEvent> g_key_events;
void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods) {
  ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);
  if (action == GLFW_PRESS) g_key_events.push_back({key, mods});
}

void hookViewportCallbacks() {
  for (ImGuiViewport *viewport : ImGui::GetPlatformIO().Viewports) {
    if (viewport->PlatformHandle == nullptr || viewport == ImGui::GetMainViewport()) continue;
    glfwSetKeyCallback((GLFWwindow *)viewport->PlatformHandle, keyCallback);
  }
}

void glfwErrorCallback(int error, const char *description) {
  fprintf(stderr, "GLFW error %d: %s\n", error, description);
}

#ifndef __EMSCRIPTEN__
void paceFrame() {
  using clock = std::chrono::steady_clock;
  static clock::duration period = [] {
    const GLFWvidmode *mode = glfwGetVideoMode(glfwGetPrimaryMonitor());
    int hz = (mode != nullptr && mode->refreshRate > 0) ? mode->refreshRate : 60;
    return std::chrono::duration_cast<clock::duration>(std::chrono::duration<double>(1.0 / hz));
  }();
  static clock::time_point next = clock::now();
  next += period;
  auto now = clock::now();
  if (next < now) {
    next = now;  // fell behind (slow frame or hidden window): don't try to catch up
    return;
  }
  std::this_thread::sleep_until(next);
}
#endif

void renderFrame(GLFWwindow *window, MainWindow *win) {
  glfwPollEvents();
  utils::drainMainThreadQueue();

  int fb_w = 0, fb_h = 0;
  glfwGetFramebufferSize(window, &fb_w, &fb_h);

  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();
  win->draw();
  ImGui::Render();

  const ImVec4 &bg = ImGui::GetStyle().Colors[ImGuiCol_WindowBg];
  glViewport(0, 0, fb_w, fb_h);
  glClearColor(bg.x, bg.y, bg.z, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT);
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

  if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
    GLFWwindow *backup_context = glfwGetCurrentContext();
    ImGui::UpdatePlatformWindows();
    hookViewportCallbacks();
    ImGui::RenderPlatformWindowsDefault();
    glfwMakeContextCurrent(backup_context);
  }
#ifndef __EMSCRIPTEN__  // the browser presents the canvas after each frame
  glfwSwapBuffers(window);
  paceFrame();
#endif
}

class GlfwRuntime {
public:
  GlfwRuntime() {
    glfwSetErrorCallback(glfwErrorCallback);
#ifdef __APPLE__
    setMacAppName("Cabana");
#endif
    if (!glfwInit()) throw std::runtime_error("glfwInit failed");
#ifdef __EMSCRIPTEN__
    // the glfw port maps the major version to the WebGL version
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
#else
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#endif
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
#endif
    // Restore geometry and render the initial layout before mapping the window.
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    window_ = glfwCreateWindow(1600, 900, "Cabana", nullptr, nullptr);
    if (window_ == nullptr) {
      glfwTerminate();
      throw std::runtime_error("glfwCreateWindow failed");
    }
    glfwMakeContextCurrent(window_);
#ifdef __EMSCRIPTEN__
    emscripten_glfw_make_canvas_resizable(window_, "window", nullptr);
#else
    glfwSwapInterval(0);
#endif
  }

  ~GlfwRuntime() {
    if (window_ != nullptr) glfwDestroyWindow(window_);
    glfwTerminate();
  }

  GlfwRuntime(const GlfwRuntime &) = delete;
  GlfwRuntime &operator=(const GlfwRuntime &) = delete;
  GLFWwindow *window() const { return window_; }

private:
  GLFWwindow *window_ = nullptr;
};

class ImGuiRuntime {
public:
  explicit ImGuiRuntime(GLFWwindow *window) {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
#ifndef __EMSCRIPTEN__
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
    io.ConfigViewportsNoDecoration = true;
#endif
    io.ConfigDockingTransparentPayload = true;
    io.IniFilename = nullptr;
    io.LogFilename = nullptr;
    if (!ImGui_ImplGlfw_InitForOpenGL(window, true)) {
      ImPlot::DestroyContext();
      ImGui::DestroyContext();
      throw std::runtime_error("ImGui_ImplGlfw_InitForOpenGL failed");
    }
    glfwSetKeyCallback(window, keyCallback);
#ifdef __EMSCRIPTEN__
    const char *glsl_version = "#version 300 es";
#else
    const char *glsl_version = "#version 330";
#endif
    if (!ImGui_ImplOpenGL3_Init(glsl_version)) {
      ImGui_ImplGlfw_Shutdown();
      ImPlot::DestroyContext();
      ImGui::DestroyContext();
      throw std::runtime_error("ImGui_ImplOpenGL3_Init failed");
    }
  }

  ~ImGuiRuntime() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
  }

  ImGuiRuntime(const ImGuiRuntime &) = delete;
  ImGuiRuntime &operator=(const ImGuiRuntime &) = delete;
};

}  // namespace

std::vector<KeyEvent> takeKeyEvents() {
  return std::exchange(g_key_events, {});
}

// everything that lives for the duration of the app, in construction order
struct App {
  App(std::unique_ptr<AbstractStream> stream, StreamLoader stream_loader, const std::string &dbc_file)
      : glfw(), imgui(glfw.window()), win(nullptr) {
    loadFonts();
    applyTheme(settings.theme);
    inistate::addSettingsHandler();
    inistate::load();
    inistate::applyWindowGeometry(glfw.window());
    win = std::make_unique<MainWindow>(glfw.window(), std::move(stream), std::move(stream_loader), dbc_file);
  }

  void step() {
    if (g_signal_exit.exchange(false)) {
      printf("\nexiting...\n");
      win->close();
    } else if (glfwWindowShouldClose(glfw.window())) {
      glfwSetWindowShouldClose(glfw.window(), GLFW_FALSE);
      win->close();
    }
    renderFrame(glfw.window(), win.get());
  }

  GlfwRuntime glfw;
  ImGuiRuntime imgui;
  std::unique_ptr<MainWindow> win;
};

#ifdef __EMSCRIPTEN__
int run(std::unique_ptr<AbstractStream> stream, StreamLoader stream_loader, const std::string &dbc_file) {
  App *app = nullptr;
  try {
    app = new App(std::move(stream), std::move(stream_loader), dbc_file);
  } catch (const std::exception &e) {
    fprintf(stderr, "%s\n", e.what());
    return 1;
  }
  glfwShowWindow(app->glfw.window());
  // the browser owns the loop; the page going away is the only exit
  emscripten_set_main_loop_arg([](void *arg) { static_cast<App *>(arg)->step(); }, app, 0, false);
  return 0;
}
#else
int run(std::unique_ptr<AbstractStream> stream, StreamLoader stream_loader, const std::string &dbc_file) {
  try {
    // SIGINT/SIGTERM close all windows (which may ask about unsaved changes), then exit
    UnixSignalHandler signal_handler([]() { g_signal_exit = true; });
    App app(std::move(stream), std::move(stream_loader), dbc_file);
    renderFrame(app.glfw.window(), app.win.get());
    glfwShowWindow(app.glfw.window());
    while (!app.win->exited()) {
      app.step();
    }
    return 0;
  } catch (const std::exception &e) {
    fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
#endif
