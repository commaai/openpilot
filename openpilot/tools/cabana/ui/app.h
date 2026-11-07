#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "tools/cabana/streams/abstractstream.h"

// builds a stream off the main thread; nullptr means the load failed
using StreamLoader = std::function<std::unique_ptr<AbstractStream>()>;

// takes ownership of stream; a loader runs behind the window instead of before it; with neither, the
// stream selector opens
int run(std::unique_ptr<AbstractStream> stream, StreamLoader stream_loader, const std::string &dbc_file);

// key presses with the modifier state at event time: imgui may apply a modifier release in the same frame as
// the key press it belongs to, which loses fast shortcut sequences. Consumed once per frame by MainWindow.
struct KeyEvent {
  int key;   // GLFW_KEY_*
  int mods;  // GLFW_MOD_*
};
std::vector<KeyEvent> takeKeyEvents();
