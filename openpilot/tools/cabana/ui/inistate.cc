#include "tools/cabana/ui/inistate.h"

#include <cstdio>
#include <cstring>

#include "imgui.h"
#include "imgui_internal.h"
#include <GLFW/glfw3.h>

#include "tools/cabana/settings.h"
#include "tools/cabana/ui/util.h"

namespace inistate {

MainWindowState main_window;

namespace {

void *readOpen(ImGuiContext *, ImGuiSettingsHandler *, const char *name) {
  return strcmp(name, "MainWindow") == 0 ? (void *)&main_window : nullptr;
}

void readLine(ImGuiContext *, ImGuiSettingsHandler *, void *entry, const char *line) {
  auto *state = (MainWindowState *)entry;
  int x = 0, y = 0, flag = 0;
  float ratio = 0.0f;
  if (sscanf(line, "Pos=%d,%d", &x, &y) == 2) {
    state->pos[0] = x;
    state->pos[1] = y;
  } else if (sscanf(line, "Size=%d,%d", &x, &y) == 2) {
    state->size[0] = x;
    state->size[1] = y;
    state->has_geometry = true;
  } else if (sscanf(line, "Maximized=%d", &flag) == 1) {
    state->maximized = flag != 0;
  } else if (sscanf(line, "VideoSplitterRatio=%f", &ratio) == 1) {
    state->video_splitter_ratio = ratio;
  } else if (sscanf(line, "MessagesVisible=%d", &flag) == 1) {
    state->messages_visible = flag != 0;
  } else if (sscanf(line, "ChartsVisible=%d", &flag) == 1) {
    state->charts_visible = flag != 0;
  } else if (sscanf(line, "VideoVisible=%d", &flag) == 1) {
    state->video_visible = flag != 0;
  }
}

void writeAll(ImGuiContext *, ImGuiSettingsHandler *handler, ImGuiTextBuffer *buf) {
  buf->appendf("[%s][MainWindow]\n", handler->TypeName);
  if (main_window.has_geometry) {
    buf->appendf("Pos=%d,%d\n", main_window.pos[0], main_window.pos[1]);
    buf->appendf("Size=%d,%d\n", main_window.size[0], main_window.size[1]);
  }
  buf->appendf("Maximized=%d\n", main_window.maximized ? 1 : 0);
  buf->appendf("VideoSplitterRatio=%.4f\n", main_window.video_splitter_ratio);
  buf->appendf("MessagesVisible=%d\n", main_window.messages_visible ? 1 : 0);
  buf->appendf("VideoVisible=%d\n", main_window.video_visible ? 1 : 0);
  buf->appendf("ChartsVisible=%d\n", main_window.charts_visible ? 1 : 0);
  buf->append("\n");
}

void migrateDockLayout() {
  // Show dock tabs hidden by older layouts.
  if (const auto *center = ImGui::FindWindowSettingsByID(ImHashStr("###CenterWidget"))) {
    if (auto *node = ImGui::DockBuilderGetNode(center->DockId)) {
      node->LocalFlags &= ~ImGuiDockNodeFlags_NoTabBar;
    }
  }
}

}  // namespace

void addSettingsHandler() {
  ImGuiSettingsHandler handler;
  handler.TypeName = "Cabana";
  handler.TypeHash = ImHashStr("Cabana");
  handler.ReadOpenFn = readOpen;
  handler.ReadLineFn = readLine;
  handler.WriteAllFn = writeAll;
  ImGui::AddSettingsHandler(&handler);
}

void load() {
  if (!settings.ui_state.empty())
    ImGui::LoadIniSettingsFromMemory(settings.ui_state.data(), settings.ui_state.size());

  migrateDockLayout();
}

void applyWindowGeometry(GLFWwindow *window) {
  if (main_window.has_geometry && main_window.size[0] > 0 && main_window.size[1] > 0) {
    glfwSetWindowPos(window, main_window.pos[0], main_window.pos[1]);
    glfwSetWindowSize(window, main_window.size[0], main_window.size[1]);
  }
  if (main_window.maximized) glfwMaximizeWindow(window);
}

std::string save() {
  return std::string(ImGui::SaveIniSettingsToMemory());
}

}  // namespace inistate
