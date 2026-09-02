#include "tools/cabana/ui/inistate.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

#include "imgui.h"
#include "imgui_internal.h"
#include <GLFW/glfw3.h>

#include "tools/cabana/settings.h"
#include "tools/cabana/ui/qtstate.h"
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
  buf->append("\n");
}

std::string migrateQtHeaderState(const qtstate::QtHeaderState &header) {
  // imgui restarts the hash at "###" in a window name and BeginTable seeds the table id from the window id
  const ImGuiID table_id = ImHashStr("messages", 0, ImHashStr(MESSAGES_PANEL_ID, 0));
  const float cell_padding = ImGui::GetStyle().CellPadding.x;

  ImGuiTextBuffer buf;
  buf.appendf("[Table][0x%08X,%d]\n", table_id, qtstate::kMessageColumnCount);
  for (int i = 0; i < qtstate::kMessageColumnCount; ++i) {
    buf.appendf("Column %-2d", i);
    if (i == 6) {
      buf.append(" Weight=1.0000");  // DATA is the stretch column
    } else {
      buf.appendf(" Width=%d", std::max(1, (int)(header.width[i] - 2 * cell_padding)));
    }
    buf.appendf(" Visible=%d Order=%d", header.hidden[i] ? 0 : 1, header.visual[i]);
    if (header.sort_shown && i == header.sort_section) {
      // the port feeds imgui the flipped direction so the arrow matches Qt (flipSortDirection
      // in ui/widgets/messageswidget.cc): Qt ascending is imgui descending
      buf.appendf(" Sort=0%c", header.sort_order == 0 ? '^' : 'v');
    }
    buf.append("\n");
  }
  buf.append("\n");
  return std::string(buf.c_str());
}

std::string migrateQtState() {
  auto geometry = qtstate::parseQtGeometry(settings.geometry);
  auto splitter = qtstate::parseQtSplitter(settings.video_splitter_state);

  ImGuiTextBuffer buf;
  if (geometry || splitter) {
    buf.append("[Cabana][MainWindow]\n");
    if (geometry) {
      buf.appendf("Pos=%d,%d\n", geometry->x, geometry->y);
      buf.appendf("Size=%d,%d\n", geometry->w, geometry->h);
      buf.appendf("Maximized=%d\n", geometry->maximized ? 1 : 0);
    }
    if (splitter) buf.appendf("VideoSplitterRatio=%.4f\n", splitter->ratio);
    buf.append("\n");
  }
  if (auto header = qtstate::parseQtHeaderState(settings.message_header_state)) {
    buf.append(migrateQtHeaderState(*header).c_str());
  }
  return std::string(buf.c_str());
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
  if (settings.ui_state.empty()) settings.ui_state = migrateQtState();
  if (!settings.ui_state.empty())
    ImGui::LoadIniSettingsFromMemory(settings.ui_state.data(), settings.ui_state.size());
}

void applyWindowGeometry(GLFWwindow *window) {
  // Qt restoreGeometry corrects off-screen geometry, here we rely on the window manager
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
