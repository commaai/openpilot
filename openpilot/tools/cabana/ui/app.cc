#include "tools/cabana/ui/app.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <utility>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_opengl3_loader.h"
#include "implot.h"
#include <GLFW/glfw3.h>

#include "tools/cabana/settings.h"
#include "tools/cabana/ui/mainwin.h"
#include "tools/cabana/utils/util.h"

namespace {

std::atomic<bool> g_signal_exit{false};
std::vector<KeyEvent> g_key_events;
// Qt repaints on demand; imgui needs a few frames after an input event for hover/active state to settle
int g_input_frames = 0;
constexpr int SETTLE_FRAMES = 3;

void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods) {
  ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);
  if (action == GLFW_PRESS) g_key_events.push_back({key, mods});
  g_input_frames = SETTLE_FRAMES;
}
void cursorPosCallback(GLFWwindow *w, double x, double y) { ImGui_ImplGlfw_CursorPosCallback(w, x, y); g_input_frames = SETTLE_FRAMES; }
void cursorEnterCallback(GLFWwindow *w, int entered) { ImGui_ImplGlfw_CursorEnterCallback(w, entered); g_input_frames = SETTLE_FRAMES; }
void mouseButtonCallback(GLFWwindow *w, int b, int a, int m) { ImGui_ImplGlfw_MouseButtonCallback(w, b, a, m); g_input_frames = SETTLE_FRAMES; }
void scrollCallback(GLFWwindow *w, double x, double y) { ImGui_ImplGlfw_ScrollCallback(w, x, y); g_input_frames = SETTLE_FRAMES; }
void charCallback(GLFWwindow *w, unsigned int c) { ImGui_ImplGlfw_CharCallback(w, c); g_input_frames = SETTLE_FRAMES; }
void windowFocusCallback(GLFWwindow *w, int f) { ImGui_ImplGlfw_WindowFocusCallback(w, f); g_input_frames = SETTLE_FRAMES; }
void windowSizeCallback(GLFWwindow *, int, int) { g_input_frames = SETTLE_FRAMES; }
void windowRefreshCallback(GLFWwindow *) { g_input_frames = SETTLE_FRAMES; }

// Paces the loop like Qt's event driven repaints: input renders right away, a playing stream repaints at the
// rate its data and camera frames arrive, otherwise the window idles.
double frameInterval() {
  if (g_input_frames > 0) return 0.0;
  const bool playing = can != nullptr && dynamic_cast<DummyStream *>(can) == nullptr && !can->isPaused();
  return playing ? 1.0 / 30.0 : 1.0 / 10.0;
}

void glfwErrorCallback(int error, const char *description) {
  fprintf(stderr, "GLFW error %d: %s\n", error, description);
}

void renderFrame(GLFWwindow *window, MainWindow *win, double wait_seconds) {
  if (wait_seconds > 0) {
    glfwWaitEventsTimeout(wait_seconds);
  } else {
    glfwPollEvents();
  }
  if (g_input_frames > 0) --g_input_frames;
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
  glfwSwapBuffers(window);
}

}  // namespace

GlfwRuntime::GlfwRuntime() {
  glfwSetErrorCallback(glfwErrorCallback);
  if (!glfwInit()) throw std::runtime_error("glfwInit failed");
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
#endif
  window_ = glfwCreateWindow(1600, 900, "Cabana", nullptr, nullptr);
  if (window_ == nullptr) {
    glfwTerminate();
    throw std::runtime_error("glfwCreateWindow failed");
  }
  glfwMakeContextCurrent(window_);
  glfwSwapInterval(1);
}

GlfwRuntime::~GlfwRuntime() {
  if (window_ != nullptr) glfwDestroyWindow(window_);
  glfwTerminate();
}

ImGuiRuntime::ImGuiRuntime(GLFWwindow *window) {
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImPlot::CreateContext();
  ImGuiIO &io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
  io.IniFilename = nullptr;
  io.LogFilename = nullptr;
  if (!ImGui_ImplGlfw_InitForOpenGL(window, true)) {
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    throw std::runtime_error("ImGui_ImplGlfw_InitForOpenGL failed");
  }
  // chain the imgui backend callbacks so input marks the next frames dirty
  glfwSetKeyCallback(window, keyCallback);
  glfwSetCursorPosCallback(window, cursorPosCallback);
  glfwSetCursorEnterCallback(window, cursorEnterCallback);
  glfwSetMouseButtonCallback(window, mouseButtonCallback);
  glfwSetScrollCallback(window, scrollCallback);
  glfwSetCharCallback(window, charCallback);
  glfwSetWindowFocusCallback(window, windowFocusCallback);
  glfwSetWindowSizeCallback(window, windowSizeCallback);
  glfwSetWindowRefreshCallback(window, windowRefreshCallback);
  if (!ImGui_ImplOpenGL3_Init("#version 330")) {
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    throw std::runtime_error("ImGui_ImplOpenGL3_Init failed");
  }
}

ImGuiRuntime::~ImGuiRuntime() {
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImPlot::DestroyContext();
  ImGui::DestroyContext();
}

std::vector<KeyEvent> takeKeyEvents() {
  return std::exchange(g_key_events, {});
}

int run(std::unique_ptr<AbstractStream> stream, const std::string &dbc_file) {
  try {
    // SIGINT/SIGTERM close all windows (the close flow may ask about unsaved changes), then exit
    UnixSignalHandler signal_handler([]() { g_signal_exit = true; });

    GlfwRuntime glfw;
    ImGuiRuntime imgui(glfw.window());
    loadFonts();
    applyTheme(settings.theme);

    MainWindow win(glfw.window(), std::move(stream), dbc_file);
    auto last_frame = std::chrono::steady_clock::now();
    while (!win.exited()) {
      if (g_signal_exit.exchange(false)) {
        printf("\nexiting...\n");
        win.close();
      } else if (glfwWindowShouldClose(glfw.window())) {
        glfwSetWindowShouldClose(glfw.window(), GLFW_FALSE);
        win.close();
      }
      // the interval is measured from the start of the previous frame, so render time counts against it
      const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - last_frame).count();
      const double wait_seconds = std::max(0.0, frameInterval() - elapsed);
      last_frame = std::chrono::steady_clock::now() + std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(wait_seconds));
      renderFrame(glfw.window(), &win, wait_seconds);
    }
    return 0;
  } catch (const std::exception &e) {
    fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
