#pragma once

#include <memory>
#include <string>
#include <vector>

#include "tools/cabana/streams/abstractstream.h"

struct GLFWwindow;

// takes ownership of stream; nullptr opens the stream selector
int run(std::unique_ptr<AbstractStream> stream, const std::string &dbc_file);

class GlfwRuntime {
public:
  GlfwRuntime();
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

// key presses with the modifier state at event time: imgui may apply a modifier release in the same frame as
// the key press it belongs to, which loses fast shortcut sequences. Consumed once per frame by MainWindow.
struct KeyEvent {
  int key;   // GLFW_KEY_*
  int mods;  // GLFW_MOD_*
};
std::vector<KeyEvent> takeKeyEvents();

// style.cc
void loadFonts();
void applyTheme(int theme);  // safe to call at runtime
bool isDarkTheme();  // the theme applyTheme() resolved, so AUTO_THEME reports what is on screen
