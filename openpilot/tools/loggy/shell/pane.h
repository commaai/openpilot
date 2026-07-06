#pragma once

#include <cstddef>
#include <string_view>

namespace loggy {

struct PaneInstance;
struct Session;

// Pane draw functions run on the UI thread during a single frame. They may keep
// local UI state in PaneInstance::state_json, but borrowed backend/store views
// must not escape the frame that handed them out.
using PaneDrawFn = void (*)(Session &, PaneInstance &);

struct PaneType {
  const char *id = "";
  const char *display_name = "";
  PaneDrawFn draw = nullptr;
};

const PaneType *pane_type(std::string_view id);
const PaneType *pane_types();
size_t pane_type_count();

}  // namespace loggy
