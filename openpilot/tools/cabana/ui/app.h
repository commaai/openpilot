#pragma once

#include <memory>
#include <string>
#include <vector>

#include "tools/cabana/streams/abstractstream.h"

// takes ownership of stream; nullptr opens the stream selector
int run(std::unique_ptr<AbstractStream> stream, const std::string &dbc_file);

// key presses with the modifier state at event time: imgui may apply a modifier release in the same frame as
// the key press it belongs to, which loses fast shortcut sequences. Consumed once per frame by MainWindow.
struct KeyEvent {
  int key;   // GLFW_KEY_*
  int mods;  // GLFW_MOD_*
};
std::vector<KeyEvent> takeKeyEvents();
