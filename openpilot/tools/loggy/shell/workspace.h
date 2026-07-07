#pragma once

#include "tools/loggy/shell/pane.h"

#include <any>
#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace loggy {

inline constexpr const char *kDefaultPaneType = "empty";
inline constexpr const char *kDefaultPaneTitle = "...";

struct PaneInstance {
  PaneInstance() = default;
  PaneInstance(const PaneInstance &other)
    : type(other.type),
      title(other.title),
      state_json(other.state_json),
      selection_group(other.selection_group) {}
  PaneInstance &operator=(const PaneInstance &other) {
    if (this == &other) return *this;
    type = other.type;
    title = other.title;
    state_json = other.state_json;
    selection_group = other.selection_group;
    transient_state.reset();
    return *this;
  }
  PaneInstance(PaneInstance &&) noexcept = default;
  PaneInstance &operator=(PaneInstance &&) noexcept = default;

  std::string type = kDefaultPaneType;
  std::string title = kDefaultPaneTitle;
  std::string state_json = "{}";
  std::string selection_group = "default";
  std::any transient_state;  // Per-pane draw caches; not serialized into workspace JSON.
};

enum class SplitOrientation {
  Horizontal,
  Vertical,
};

enum class PaneSplit {
  Left,
  Right,
  Top,
  Bottom,
};

struct WorkspaceNode {
  bool is_pane = false;
  int pane_index = -1;
  SplitOrientation orientation = SplitOrientation::Horizontal;
  std::vector<float> sizes;
  std::vector<WorkspaceNode> children;
};

struct WorkspaceTab {
  std::string name = "tab1";
  WorkspaceNode root;
  std::vector<PaneInstance> panes;
};

struct Workspace {
  std::vector<WorkspaceTab> tabs;
  int current_tab_index = 0;
};

// The outer frame identity: a fixed left dock (the browser for jotpluggler, the message list for
// cabana) plus which saved layouts belong to it. The dock lives OUTSIDE the Workspace, so loading
// a layout can never remove it — content workspaces and every saved layout are chrome-free by the
// normalize_workspace() invariant. This is what makes cabana/jotpluggler "core shells".
struct Shell {
  std::string kind = "loggy";  // "cabana" | "jotpluggler" | "loggy"
  PaneInstance dock;           // the fixed chrome pane (browser or messages)
  bool dock_open = true;
  float dock_width = 320.0f;
};

// Chrome pane types belong only in a shell dock, never in a content workspace or a saved layout.
bool is_shell_chrome_pane_type(std::string_view type);
std::string shell_kind_for_preset(std::string_view preset);
Shell make_shell(std::string_view kind);

struct WorkspaceHistory {
  static constexpr size_t kMaxHistory = 50;

  void reset(const Workspace &workspace);
  void push(const Workspace &workspace);
  bool can_undo() const;
  bool can_redo() const;
  const Workspace *undo();
  const Workspace *redo();

  std::vector<Workspace> history;
  int position = -1;
};

struct WorkspaceLoadResult {
  Workspace workspace;
  bool loaded_draft = false;
};

bool workspace_autosave_available(const std::filesystem::path &layout_path);
void autosave_workspace(const Workspace &workspace, const std::filesystem::path &layout_path,
                       std::string &workspace_status, std::string_view status);
void record_workspace_change(Workspace &workspace, WorkspaceHistory &history, const std::filesystem::path &layout_path,
                            std::string &workspace_status, std::string_view status);
std::optional<int> restore_workspace_snapshot(Workspace &workspace, const Workspace *snapshot,
                                             const std::filesystem::path &layout_path, std::string &workspace_status,
                                             std::string_view status);
void save_workspace_now(Workspace &workspace, WorkspaceHistory &history, const std::filesystem::path &layout_path,
                       std::string &workspace_status, std::string_view shell_kind = {});
void clear_workspace_draft_now(const std::filesystem::path &layout_path, std::string &workspace_status);

PaneInstance make_pane(std::string type = kDefaultPaneType,
                       std::string title = kDefaultPaneTitle,
                       std::string state_json = "{}");
WorkspaceTab make_tab(std::string name, PaneInstance pane = make_pane());
Workspace make_empty_workspace();
Workspace make_cabana_workspace();
Workspace make_jotpluggler_workspace();
Workspace make_default_workspace(std::string_view preset);

WorkspaceTab *active_tab(Workspace *workspace);
const WorkspaceTab *active_tab(const Workspace &workspace);

std::string next_tab_name(const Workspace &workspace, std::string_view base_name = "tab1");
int add_tab(Workspace *workspace, std::string name = {});

bool split_pane(WorkspaceTab *tab, int pane_index, PaneSplit split, PaneInstance pane = make_pane());
bool close_pane(WorkspaceTab *tab, int pane_index);

void normalize_workspace(Workspace *workspace);

std::string workspace_to_json(const Workspace &workspace, std::string_view shell_kind = {});
Workspace workspace_from_json(std::string_view json_text, const std::filesystem::path &source = {});
void save_workspace_json(const Workspace &workspace, const std::filesystem::path &path, std::string_view shell_kind = {});
Workspace load_workspace_json(const std::filesystem::path &path);

std::filesystem::path layouts_dir();
std::vector<std::string> available_layout_names();
// Saved layouts whose "shell" tag matches the given shell (untagged layouts show in every shell).
std::vector<std::string> layouts_for_shell(std::string_view kind);
// The "shell" tag written in a layout file, or empty if it has none / can't be read.
std::string layout_shell_tag(const std::filesystem::path &path);
std::filesystem::path autosave_dir();
std::filesystem::path autosave_path_for_layout(const std::filesystem::path &layout_path);
void save_workspace_draft(const Workspace &workspace, const std::filesystem::path &layout_path);
WorkspaceLoadResult load_workspace_or_draft(const std::filesystem::path &layout_path);
void clear_workspace_draft(const std::filesystem::path &layout_path);

}  // namespace loggy
