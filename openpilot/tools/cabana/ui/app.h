#pragma once

#include <memory>
#include <string>

#include "imgui.h"
#include "tools/cabana/core/settings.h"
#include "tools/cabana/streams/abstractstream.h"

struct GLFWwindow;

struct Options {
  bool demo = false;
  bool auto_source = false;
  bool qcam = false;
  bool wide_road = false;
  bool cabin = false;
  bool msgq = false;
  bool panda = false;
  bool no_vipc = false;
  std::string panda_serial;
  std::string socketcan;
  std::string zmq;
  std::string data_dir;
  std::string dbc;
  std::string route;
};

int run(const Options &options);

// nullptr with an empty *error means no stream was requested (open the stream selector)
std::unique_ptr<AbstractStream> createStream(const Options &options, std::string *error);

class GlfwRuntime {
public:
  explicit GlfwRuntime(const char *title);
  ~GlfwRuntime();
  GlfwRuntime(const GlfwRuntime &) = delete;
  GlfwRuntime &operator=(const GlfwRuntime &) = delete;
  GLFWwindow *window() const { return window_; }

private:
  GLFWwindow *window_ = nullptr;
};

class ImGuiRuntime {
public:
  explicit ImGuiRuntime(GLFWwindow *window);
  ~ImGuiRuntime();
  ImGuiRuntime(const ImGuiRuntime &) = delete;
  ImGuiRuntime &operator=(const ImGuiRuntime &) = delete;
};

// style.cc
void loadFonts();
void applyTheme(int theme);
void pushMonoFont();
void popMonoFont();
void pushBoldFont();
void popBoldFont();
ImVec4 colorRgb(int r, int g, int b, float alpha = 1.0f);

struct UiState {
  bool show_messages = true;
  bool show_detail = true;
  bool show_video = true;
  bool show_charts = true;
  bool show_fps = false;
  bool reset_layout = true;
  bool request_close = false;
  bool open_stream_selector = false;
  bool open_settings = false;
  CabanaSettingsState settings_draft;
  MessageId selected_id;
  std::string status_text;
  double status_until = 0;
  std::string error_text;
};

struct App {
  UiState ui;
  std::unique_ptr<AbstractStream> stream;  // `can` points at this (or at dummy)
  DummyStream dummy;
  Connections connections;
  std::string route_input;
  std::string data_dir_input;
  bool has_stream() const { return stream != nullptr; }
};

// layout.cc
void drawFrame(App *app);
void showStatus(App *app, const std::string &text, double seconds = 2.0);

// app.cc
void startStream(App *app, std::unique_ptr<AbstractStream> stream, const std::string &dbc_file);
void closeStream(App *app);
