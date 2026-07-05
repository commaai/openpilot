#pragma once

#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace loggy {

struct PaneInstance;
struct Session;

// Pane draw functions run on the UI thread during a single frame. They may keep
// local UI state in PaneInstance::state_json, but borrowed backend/store views
// must not escape the frame that handed them out.
using PaneDrawFn = std::function<void(Session &, PaneInstance &)>;

struct PaneType {
  std::string id;
  std::string display_name;
  PaneDrawFn draw;
};

class PaneRegistry {
public:
  bool registerType(PaneType type);
  const PaneType *find(std::string_view id) const;
  std::vector<PaneType> types() const;
  void clear();

private:
  std::vector<PaneType> types_;
};

PaneRegistry &pane_registry();
void register_dummy_pane_types(PaneRegistry &registry = pane_registry());

}  // namespace loggy
