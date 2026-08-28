#pragma once

#include <memory>
#include <string>

#include "tools/cabana/streams/abstractstream.h"

struct GLFWwindow;

// takes ownership of stream (nullptr opens the stream selector), mirrors `MainWindow w(stream, dbc); app.exec()`
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

// style.cc
void loadFonts();
void applyTheme(int theme);  // utils::setTheme equivalent, safe to call at runtime
