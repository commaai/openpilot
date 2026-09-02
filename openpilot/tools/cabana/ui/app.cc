#include "tools/cabana/ui/app.h"

#include <atomic>
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
// imgui releases every mouse button when the window loses focus, which aborts a panel tear-off drag and
// docks the panel back. X11 keeps delivering the drag through the implicit grab, so hold a focus loss back
// while a button is down and deliver it after the release (see deliverPendingFocusLoss).
GLFWwindow *g_focus_lost_window = nullptr;
// macOS drops the button on its own when the focus moves, and holding the loss back there swallowed the
// first click in a popup: the click makes the popup's window key, the main window's loss lands on the
// release and imgui clears its mouse state before it sees that release
void windowFocusCallback(GLFWwindow *w, int f) {
#ifdef __APPLE__
  ImGui_ImplGlfw_WindowFocusCallback(w, f);
#else
  if (f) {
    g_focus_lost_window = nullptr;
    ImGui_ImplGlfw_WindowFocusCallback(w, f);
  } else {
    g_focus_lost_window = w;
  }
#endif
}
bool anyMouseButtonDown(GLFWwindow *w) {
  for (int b = GLFW_MOUSE_BUTTON_1; b <= GLFW_MOUSE_BUTTON_LAST; ++b) {
    if (glfwGetMouseButton(w, b) == GLFW_PRESS) return true;
  }
  return false;
}
void deliverPendingFocusLoss() {
  if (g_focus_lost_window == nullptr || anyMouseButtonDown(g_focus_lost_window)) return;
  ImGui_ImplGlfw_WindowFocusCallback(g_focus_lost_window, GLFW_FALSE);
  g_focus_lost_window = nullptr;
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

// vsync paces the loop: glfwSwapBuffers blocks until the next refresh. Throttling on top of that beats
// against the refresh rate and makes the camera view stutter.
void renderFrame(GLFWwindow *window, MainWindow *win) {
  glfwPollEvents();
  deliverPendingFocusLoss();
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
  glfwSwapBuffers(window);
}

class GlfwRuntime {
public:
  GlfwRuntime() {
    glfwSetErrorCallback(glfwErrorCallback);
#ifdef __APPLE__
    setMacAppName("Cabana");
#endif
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
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
    io.ConfigViewportsNoDecoration = false;
    io.IniFilename = nullptr;
    io.LogFilename = nullptr;
    if (!ImGui_ImplGlfw_InitForOpenGL(window, true)) {
      ImPlot::DestroyContext();
      ImGui::DestroyContext();
      throw std::runtime_error("ImGui_ImplGlfw_InitForOpenGL failed");
    }
    glfwSetKeyCallback(window, keyCallback);
    glfwSetWindowFocusCallback(window, windowFocusCallback);
    if (!ImGui_ImplOpenGL3_Init("#version 330")) {
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

int run(std::unique_ptr<AbstractStream> stream, StreamLoader stream_loader, const std::string &dbc_file) {
  try {
    // SIGINT/SIGTERM close all windows (which may ask about unsaved changes), then exit
    UnixSignalHandler signal_handler([]() { g_signal_exit = true; });

    GlfwRuntime glfw;
    ImGuiRuntime imgui(glfw.window());
    loadFonts();
    applyTheme(settings.theme);
    inistate::addSettingsHandler();
    inistate::load();
    inistate::applyWindowGeometry(glfw.window());

    MainWindow win(glfw.window(), std::move(stream), std::move(stream_loader), dbc_file);
    while (!win.exited()) {
      if (g_signal_exit.exchange(false)) {
        printf("\nexiting...\n");
        win.close();
      } else if (glfwWindowShouldClose(glfw.window())) {
        glfwSetWindowShouldClose(glfw.window(), GLFW_FALSE);
        win.close();
      }
      renderFrame(glfw.window(), &win);
    }
    return 0;
  } catch (const std::exception &e) {
    fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
