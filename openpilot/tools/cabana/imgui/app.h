#pragma once

#include <filesystem>
#include <string>

#include "imgui.h"

struct GLFWwindow;

struct Options {
  // stream selection (parity with tools/cabana/cabana.cc; wired up in later phases)
  std::string route;
  std::string data_dir;
  std::string dbc_path;
  std::string panda_serial;
  std::string socketcan_device;
  std::string zmq_address;
  bool msgq = false;
  bool panda = false;
  bool auto_source = false;
  bool qcam = false;
  bool ecam = false;
  bool dcam = false;
  bool no_vipc = false;

  // window / capture
  int width = 1600;
  int height = 900;
  bool show = true;
  bool dark_theme = false;
  std::string output_path;
};

enum class Theme {
  Light,
  Dark,
};

class GlfwRuntime {
public:
  explicit GlfwRuntime(const Options &options);
  ~GlfwRuntime();
  GLFWwindow *window() const { return window_; }

private:
  GLFWwindow *window_ = nullptr;
};

class ImGuiRuntime {
public:
  explicit ImGuiRuntime(GLFWwindow *window);
  ~ImGuiRuntime();
};

void save_framebuffer_png(const std::filesystem::path &output_path, int width, int height);

// theme.cc
const std::filesystem::path &repo_root();
void load_fonts();
void apply_theme(Theme theme);
ImVec4 theme_clear_color();
void push_bold_font(float size = 0.0f);
void pop_bold_font();
void push_mono_font(float size = 0.0f);
void pop_mono_font();

// app.cc
int run(const Options &options);
