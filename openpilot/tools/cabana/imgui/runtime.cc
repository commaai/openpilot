#include "tools/cabana/imgui/app.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_opengl3_loader.h"
#include "implot.h"

#include <GLFW/glfw3.h>

namespace fs = std::filesystem;

namespace {

void glfw_error_callback(int error, const char *description) {
  std::cerr << "GLFW error " << error << ": " << (description != nullptr ? description : "unknown") << "\n";
}

std::string shell_quote(const std::string &value) {
  std::string quoted = "'";
  for (char c : value) {
    if (c == '\'') {
      quoted += "'\\''";
    } else {
      quoted.push_back(c);
    }
  }
  quoted.push_back('\'');
  return quoted;
}

}  // namespace

GlfwRuntime::GlfwRuntime(const Options &options) {
  glfwSetErrorCallback(glfw_error_callback);
  if (!glfwInit()) throw std::runtime_error("glfwInit failed");

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
#endif
  const bool fixed_size = !options.show;
  glfwWindowHint(GLFW_RESIZABLE, fixed_size ? GLFW_FALSE : GLFW_TRUE);
  glfwWindowHint(GLFW_VISIBLE, options.show ? GLFW_TRUE : GLFW_FALSE);

  window_ = glfwCreateWindow(options.width, options.height, "Cabana", nullptr, nullptr);
  if (window_ == nullptr) {
    glfwTerminate();
    throw std::runtime_error("glfwCreateWindow failed");
  }

  if (fixed_size) {
    glfwSetWindowSizeLimits(window_, options.width, options.height, options.width, options.height);
  }
  glfwMakeContextCurrent(window_);
  glfwSwapInterval(options.show ? 1 : 0);
}

GlfwRuntime::~GlfwRuntime() {
  if (window_ != nullptr) {
    glfwDestroyWindow(window_);
  }
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

void save_framebuffer_png(const fs::path &output_path, int width, int height) {
  if (width <= 0 || height <= 0) throw std::runtime_error("Invalid framebuffer size");
  if (output_path.has_parent_path()) {
    fs::create_directories(output_path.parent_path());
  }

  std::vector<uint8_t> pixels(static_cast<size_t>(width) * static_cast<size_t>(height) * 4U, 0);
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());

  const fs::path ppm_path = output_path.parent_path() / (output_path.stem().string() + ".ppm");
  char header[64];
  const int header_len = snprintf(header, sizeof(header), "P6\n%d %d\n255\n", width, height);
  std::string ppm(header, header_len);
  ppm.reserve(ppm.size() + static_cast<size_t>(width) * static_cast<size_t>(height) * 3U);
  for (int y = height - 1; y >= 0; --y) {
    for (int x = 0; x < width; ++x) {
      const size_t index = static_cast<size_t>((y * width + x) * 4);
      ppm.append(reinterpret_cast<const char *>(&pixels[index]), 3);
    }
  }
  std::ofstream out(ppm_path, std::ios::binary);
  out.write(ppm.data(), ppm.size());
  if (!out.good()) throw std::runtime_error("Failed to write " + ppm_path.string());
  out.close();

  const std::string command = "convert " + shell_quote(ppm_path.string()) + " " + shell_quote(output_path.string());
  if (std::system(command.c_str()) != 0) throw std::runtime_error("image conversion failed: " + command);
  fs::remove(ppm_path);
}
