#include "tools/cabana/ui/app.h"

#include <atomic>
#include <cstdio>
#include <stdexcept>

#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_opengl3_loader.h"
#include "implot.h"
#include <GLFW/glfw3.h>

#include "tools/cabana/dbc/dbcmanager.h"
#include "tools/cabana/settings.h"
#include "tools/cabana/streams/devicestream.h"
#include "tools/cabana/streams/pandastream.h"
#include "tools/cabana/streams/replaystream.h"
#ifdef __linux__
#include "tools/cabana/streams/socketcanstream.h"
#endif
#include "tools/cabana/utils/util.h"

namespace {

std::atomic<bool> g_signal_exit{false};

void glfwErrorCallback(int error, const char *description) {
  fprintf(stderr, "GLFW error %d: %s\n", error, description);
}

void renderFrame(GLFWwindow *window, App *app) {
  glfwPollEvents();
  utils::drainMainThreadQueue();

  int fb_w = 0, fb_h = 0;
  glfwGetFramebufferSize(window, &fb_w, &fb_h);

  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();
  drawFrame(app);
  ImGui::Render();

  const ImVec4 &bg = ImGui::GetStyle().Colors[ImGuiCol_WindowBg];
  glViewport(0, 0, fb_w, fb_h);
  glClearColor(bg.x, bg.y, bg.z, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT);
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
  glfwSwapBuffers(window);
}

}  // namespace

GlfwRuntime::GlfwRuntime(const char *title) {
  glfwSetErrorCallback(glfwErrorCallback);
  if (!glfwInit()) throw std::runtime_error("glfwInit failed");
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
#endif
  window_ = glfwCreateWindow(1600, 900, title, nullptr, nullptr);
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

std::unique_ptr<AbstractStream> createStream(const Options &options, std::string *error) {
  error->clear();
  if (options.msgq) return std::make_unique<DeviceStream>();
  if (!options.zmq.empty()) return std::make_unique<DeviceStream>(options.zmq);
  if (options.panda || !options.panda_serial.empty()) {
    try {
      return std::make_unique<PandaStream>(PandaStreamConfig{.serial = options.panda_serial});
    } catch (const std::exception &e) {
      *error = e.what();
      return nullptr;
    }
  }
#ifdef __linux__
  if (SocketCanStream::available() && !options.socketcan.empty()) {
    return std::make_unique<SocketCanStream>(SocketCanStreamConfig{.device = options.socketcan});
  }
#endif
  std::string route = options.route;
  if (route.empty() && options.demo) route = DEMO_ROUTE;
  if (route.empty()) return nullptr;

  uint32_t flags = REPLAY_FLAG_NONE;
  if (options.wide_road) flags |= REPLAY_FLAG_WIDE_ROAD;
  if (options.qcam) flags |= REPLAY_FLAG_QCAMERA;
  if (options.cabin) flags |= REPLAY_FLAG_CABIN_CAMERA;
  if (options.no_vipc) flags |= REPLAY_FLAG_NO_VIPC;
  auto stream = std::make_unique<ReplayStream>();
  Connection err = stream->error.connect([error](const std::string &msg) { *error = msg; });
  if (!stream->loadRoute(route, options.data_dir, flags, options.auto_source)) {
    if (error->empty()) *error = "failed to load route " + route;
    return nullptr;
  }
  return stream;
}

void startStream(App *app, std::unique_ptr<AbstractStream> stream, const std::string &dbc_file) {
  closeStream(app);
  app->stream = std::move(stream);
  can = app->stream.get();
  app->connections.push_back(can->error.connect([app](const std::string &msg) { app->ui.error_text = msg; }));
  can->start();
  if (!dbc_file.empty()) {
    std::string err;
    if (!dbc()->open(SOURCE_ALL, dbc_file, &err)) app->ui.error_text = dbc_file + ": " + err;
  }
  showStatus(app, "Stream [" + can->routeName() + "] started");
}

void closeStream(App *app) {
  app->connections.clear();
  app->ui.selected_id = {};
  can = &app->dummy;
  app->stream.reset();
}

int run(const Options &options) {
  try {
    App app;
    can = &app.dummy;
    UnixSignalHandler signal_handler([]() { g_signal_exit = true; });

    std::string error;
    std::unique_ptr<AbstractStream> stream = createStream(options, &error);
    if (!error.empty()) {
      fprintf(stderr, "%s\n", error.c_str());
      return 1;
    }

    GlfwRuntime glfw("Cabana");
    ImGuiRuntime imgui(glfw.window());
    loadFonts();
    applyTheme(settings.theme);

    if (stream) {
      startStream(&app, std::move(stream), options.dbc);
    } else {
      app.ui.open_stream_selector = true;
    }

    while (!glfwWindowShouldClose(glfw.window())) {
      renderFrame(glfw.window(), &app);
      if (g_signal_exit || app.ui.request_close) {
        printf("\nexiting...\n");
        glfwSetWindowShouldClose(glfw.window(), GLFW_TRUE);
      }
    }
    closeStream(&app);
    settings.save();
    return 0;
  } catch (const std::exception &e) {
    fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
