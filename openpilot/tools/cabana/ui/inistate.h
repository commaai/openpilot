#pragma once
#include <string>

struct GLFWwindow;

// The imgui frontend's persisted state: the imgui ini text (windows, dock layout, table
// state) plus a custom [Cabana][MainWindow] section, stored as one string in Settings.
namespace inistate {

struct MainWindowState {
  int pos[2] = {0, 0};
  int size[2] = {0, 0};
  bool maximized = false;
  bool has_geometry = false;
  float video_splitter_ratio = -1.0f;  // < 0: video at its size hint
  bool messages_visible = true;
  bool video_visible = true;
};

extern MainWindowState main_window;

void addSettingsHandler();                    // register the [Cabana] ini section
void load();                                  // migrate Qt state if needed, then LoadIniSettingsFromMemory
void applyWindowGeometry(GLFWwindow *window); // glfw pos/size/maximize from main_window
std::string save();                           // SaveIniSettingsToMemory (caller fills main_window first)

}  // namespace inistate
