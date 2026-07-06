#include "tools/loggy/shell/workspace.h"

#include "tools/loggy/panes/binary.h"
#include "tools/loggy/panes/browser.h"
#include "tools/loggy/panes/camera.h"
#include "tools/loggy/panes/computed.h"
#include "tools/loggy/panes/find_bits.h"
#include "tools/loggy/panes/find_signal.h"
#include "tools/loggy/panes/historylog.h"
#include "tools/loggy/panes/logs.h"
#include "tools/loggy/panes/map.h"
#include "tools/loggy/panes/plot.h"
#include "tools/loggy/panes/signal.h"
#include "tools/loggy/panes/dbc.h"

#include "json11/json11.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace loggy {
namespace {

namespace fs = std::filesystem;

constexpr int kVersion = 1;

std::string repo_root() {
#ifdef LOGGY_REPO_ROOT
  return LOGGY_REPO_ROOT;
#else
  return fs::current_path().string();
#endif
}

std::string read_file(const fs::path &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in.good()) throw std::runtime_error("Failed to read " + path.string());
  return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

void write_file(const fs::path &path, std::string_view contents) {
  if (path.has_parent_path()) fs::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary);
  if (!out.good()) throw std::runtime_error("Failed to write " + path.string());
  out.write(contents.data(), static_cast<std::streamsize>(contents.size()));
  if (!out.good()) throw std::runtime_error("Failed to write " + path.string());
}

std::string sanitize_stem(std::string_view name) {
  std::string out;
  out.reserve(name.size());
  bool last_was_dash = false;
  for (char raw : name) {
    const unsigned char c = static_cast<unsigned char>(raw);
    if (std::isalnum(c) != 0) {
      out.push_back(static_cast<char>(std::tolower(c)));
      last_was_dash = false;
    } else if (raw == '-' || raw == '_') {
      out.push_back(raw);
      last_was_dash = false;
    } else if (!last_was_dash && !out.empty()) {
      out.push_back('-');
      last_was_dash = true;
    }
  }
  while (!out.empty() && out.back() == '-') out.pop_back();
  return out.empty() ? "untitled" : out;
}

std::vector<double> normalized_sizes(const json11::Json &sizes_json, size_t child_count) {
  std::vector<double> sizes;
  if (sizes_json.is_array()) {
    for (const json11::Json &value : sizes_json.array_items()) {
      if (value.is_number()) sizes.push_back(std::max(0.0, value.number_value()));
    }
  }
  if (sizes.size() != child_count || child_count == 0) {
    return std::vector<double>(child_count, child_count == 0 ? 0.0 : 1.0 / static_cast<double>(child_count));
  }
  const double total = std::accumulate(sizes.begin(), sizes.end(), 0.0);
  if (total <= 0.0) return std::vector<double>(child_count, 1.0 / static_cast<double>(child_count));
  for (double &size : sizes) size /= total;
  return sizes;
}

void normalize_split_node(WorkspaceNode *node) {
  if (node->is_pane) return;
  for (WorkspaceNode &child : node->children) normalize_split_node(&child);
  if (node->children.empty()) return;
  if (node->children.size() == 1) {
    *node = node->children.front();
    return;
  }
  if (node->sizes.size() != node->children.size()) {
    node->sizes.assign(node->children.size(), 1.0f / static_cast<float>(node->children.size()));
    return;
  }
  float total = 0.0f;
  for (float &size : node->sizes) {
    size = std::max(0.0f, size);
    total += size;
  }
  if (total <= 0.0f) {
    node->sizes.assign(node->children.size(), 1.0f / static_cast<float>(node->children.size()));
    return;
  }
  for (float &size : node->sizes) size /= total;
}

void decrement_pane_indices(WorkspaceNode *node, int removed_index) {
  if (node->is_pane) {
    if (node->pane_index > removed_index) --node->pane_index;
    return;
  }
  for (WorkspaceNode &child : node->children) decrement_pane_indices(&child, removed_index);
}

bool contains_pane_node(const WorkspaceNode &node, int pane_index) {
  if (node.is_pane) return node.pane_index == pane_index;
  return std::any_of(node.children.begin(), node.children.end(), [&](const WorkspaceNode &child) {
    return contains_pane_node(child, pane_index);
  });
}

void remap_pane_indices(WorkspaceNode *node, const std::vector<int> &mapping) {
  if (node->is_pane) {
    if (node->pane_index >= 0 && node->pane_index < static_cast<int>(mapping.size())) {
      node->pane_index = mapping[static_cast<size_t>(node->pane_index)];
    }
    return;
  }
  for (WorkspaceNode &child : node->children) remap_pane_indices(&child, mapping);
}

bool remove_pane_node(WorkspaceNode *node, int pane_index) {
  if (node->is_pane) return node->pane_index == pane_index;
  for (size_t i = 0; i < node->children.size();) {
    if (remove_pane_node(&node->children[i], pane_index)) {
      node->children.erase(node->children.begin() + static_cast<std::ptrdiff_t>(i));
      if (i < node->sizes.size()) node->sizes.erase(node->sizes.begin() + static_cast<std::ptrdiff_t>(i));
    } else {
      ++i;
    }
  }
  normalize_split_node(node);
  return !node->is_pane && node->children.empty();
}

bool remove_pane_from_tree(WorkspaceNode *root, int pane_index) {
  if (!contains_pane_node(*root, pane_index)) return false;
  return !remove_pane_node(root, pane_index);
}

SplitOrientation orientation_for_split(PaneSplit split) {
  return split == PaneSplit::Top || split == PaneSplit::Bottom ? SplitOrientation::Vertical : SplitOrientation::Horizontal;
}

bool split_before(PaneSplit split) {
  return split == PaneSplit::Left || split == PaneSplit::Top;
}

bool split_pane_node(WorkspaceNode *node, int target_pane_index, SplitOrientation orientation,
                     bool new_before, int new_pane_index) {
  if (node->is_pane) {
    if (node->pane_index != target_pane_index) return false;

    WorkspaceNode existing_pane;
    existing_pane.is_pane = true;
    existing_pane.pane_index = target_pane_index;

    WorkspaceNode new_pane;
    new_pane.is_pane = true;
    new_pane.pane_index = new_pane_index;

    node->is_pane = false;
    node->pane_index = -1;
    node->orientation = orientation;
    node->sizes = {0.5f, 0.5f};
    node->children.clear();
    if (new_before) {
      node->children.push_back(std::move(new_pane));
      node->children.push_back(std::move(existing_pane));
    } else {
      node->children.push_back(std::move(existing_pane));
      node->children.push_back(std::move(new_pane));
    }
    return true;
  }

  if (node->orientation == orientation) {
    for (size_t i = 0; i < node->children.size(); ++i) {
      WorkspaceNode &child = node->children[i];
      if (!child.is_pane || child.pane_index != target_pane_index) continue;

      WorkspaceNode new_pane;
      new_pane.is_pane = true;
      new_pane.pane_index = new_pane_index;

      const auto insert_at = node->children.begin() + static_cast<std::ptrdiff_t>(new_before ? i : i + 1);
      node->children.insert(insert_at, std::move(new_pane));
      node->sizes.assign(node->children.size(), 1.0f / static_cast<float>(node->children.size()));
      return true;
    }
  }

  for (WorkspaceNode &child : node->children) {
    if (split_pane_node(&child, target_pane_index, orientation, new_before, new_pane_index)) return true;
  }
  return false;
}

bool valid_pane_index(const WorkspaceTab &tab, int pane_index) {
  return pane_index >= 0 && pane_index < static_cast<int>(tab.panes.size());
}

void ensure_nonempty_tab(WorkspaceTab *tab) {
  if (!tab->panes.empty()) return;
  tab->panes.push_back(make_pane());
  tab->root = WorkspaceNode();
  tab->root.is_pane = true;
  tab->root.pane_index = 0;
}

json11::Json parse_pane_state(std::string_view text) {
  if (text.empty()) return json11::Json::object{};
  std::string err;
  json11::Json parsed = json11::Json::parse(std::string(text), err);
  if (!err.empty()) {
    return std::string(text);
  }
  return parsed;
}

PaneInstance pane_from_json(const json11::Json &json) {
  PaneInstance pane;
  pane.type = json["type"].is_string() ? json["type"].string_value() : kDefaultPaneType;
  pane.title = json["title"].is_string() && !json["title"].string_value().empty()
             ? json["title"].string_value()
             : kDefaultPaneTitle;
  pane.selection_group = json["selection_group"].is_string() && !json["selection_group"].string_value().empty()
                       ? json["selection_group"].string_value()
                       : "default";
  const json11::Json &state = json["state"];
  if (state.is_null()) {
    pane.state_json = json["state_json"].is_string() ? json["state_json"].string_value() : "{}";
  } else {
    pane.state_json = state.dump();
  }
  return pane;
}

std::string json_string_or(const json11::Json &json, std::string_view key, std::string fallback = {}) {
  const json11::Json &value = json[std::string(key)];
  return value.is_string() ? value.string_value() : std::move(fallback);
}

json11::Json jotpluggler_y_limits_json(const json11::Json &leaf) {
  const json11::Json &range = leaf["range"];
  if (!range.is_object()) return nullptr;
  json11::Json::object limits;
  if (range["bottom"].is_number()) limits["min"] = range["bottom"];
  if (range["top"].is_number()) limits["max"] = range["top"];
  return limits.empty() ? json11::Json(nullptr) : json11::Json(limits);
}

json11::Json jotpluggler_series_json(const json11::Json &curve) {
  json11::Json::object item;
  const std::string name = json_string_or(curve, "name");
  const json11::Json &custom_python = curve["custom_python"];
  const std::string linked_source = custom_python.is_object() ? json_string_or(custom_python, "linked_source") : std::string();
  item["path"] = !linked_source.empty() ? linked_source : name;
  if (!name.empty() && name != item["path"].string_value()) item["label"] = name;
  if (curve["color"].is_string()) item["color"] = curve["color"];

  const std::string transform = json_string_or(curve, "transform");
  if (!transform.empty()) {
    item["transform"] = transform;
    if (curve["derivative_dt"].is_number()) item["derivative_dt"] = curve["derivative_dt"];
    if (curve["scale"].is_number()) item["scale"] = curve["scale"];
    if (curve["offset"].is_number()) item["offset"] = curve["offset"];
  }
  if (custom_python.is_object()) item["custom_python"] = custom_python;
  return item;
}

bool is_jotpluggler_leaf(const json11::Json &json) {
  if (!json.is_object()) return false;
  return json["curves"].is_array() || json["kind"].is_string() || json["camera_view"].is_string();
}

PaneInstance jotpluggler_pane_from_json(const json11::Json &json) {
  const std::string kind = json_string_or(json, "kind");
  const std::string title = json_string_or(json, "title", "...");
  if (kind == "map") {
    return make_pane("map", title.empty() || title == "..." ? "Map" : title, "{}");
  }
  if (kind == "camera") {
    json11::Json::object state;
    const std::string camera_view = json_string_or(json, "camera_view");
    if (!camera_view.empty()) state["camera_view"] = camera_view;
    return make_pane("camera", title.empty() || title == "..." ? "Camera" : title, json11::Json(state).dump());
  }

  json11::Json::array series;
  for (const json11::Json &curve : json["curves"].array_items()) {
    if (curve.is_object()) series.push_back(jotpluggler_series_json(curve));
  }

  json11::Json::object state{{"series", series}};
  const json11::Json y_limits = jotpluggler_y_limits_json(json);
  if (y_limits.is_object()) state["y_limits"] = y_limits;
  if (json["range"].is_object()) state["jotpluggler_range"] = json["range"];
  return make_pane("plot", title.empty() ? "..." : title, json11::Json(state).dump());
}

json11::Json pane_to_json(const PaneInstance &pane) {
  json11::Json::object out = {
    {"type", pane.type.empty() ? std::string(kDefaultPaneType) : pane.type},
    {"title", pane.title.empty() ? std::string(kDefaultPaneTitle) : pane.title},
    {"state", parse_pane_state(pane.state_json)},
  };
  if (!pane.selection_group.empty() && pane.selection_group != "default") {
    out["selection_group"] = pane.selection_group;
  }
  return out;
}

WorkspaceNode node_from_json(const json11::Json &json, WorkspaceTab *tab) {
  WorkspaceNode node;
  if (!json.is_object()) return node;

  if (json["pane"].is_number()) {
    node.is_pane = true;
    node.pane_index = json["pane"].int_value();
    return node;
  }

  if (json["type"].is_string() || json["state"].is_object() || json["state_json"].is_string()) {
    node.is_pane = true;
    node.pane_index = static_cast<int>(tab->panes.size());
    tab->panes.push_back(pane_from_json(json));
    return node;
  }

  if (is_jotpluggler_leaf(json)) {
    node.is_pane = true;
    node.pane_index = static_cast<int>(tab->panes.size());
    tab->panes.push_back(jotpluggler_pane_from_json(json));
    return node;
  }

  const std::vector<json11::Json> children = json["children"].array_items();
  if (children.empty()) return node;

  node.orientation = json["split"].string_value() == "vertical" ? SplitOrientation::Vertical : SplitOrientation::Horizontal;
  const std::vector<double> sizes = normalized_sizes(json["sizes"], children.size());
  node.sizes.reserve(sizes.size());
  node.children.reserve(children.size());
  for (size_t i = 0; i < children.size(); ++i) {
    node.sizes.push_back(static_cast<float>(sizes[i]));
    node.children.push_back(node_from_json(children[i], tab));
  }
  normalize_split_node(&node);
  return node;
}

json11::Json node_to_json(const WorkspaceNode &node, const WorkspaceTab &tab) {
  if (node.is_pane) {
    if (node.pane_index < 0 || node.pane_index >= static_cast<int>(tab.panes.size())) return nullptr;
    json11::Json::object out = pane_to_json(tab.panes[static_cast<size_t>(node.pane_index)]).object_items();
    out["pane"] = node.pane_index;
    return out;
  }

  json11::Json::array sizes;
  for (size_t i = 0; i < node.children.size(); ++i) {
    sizes.push_back(i < node.sizes.size() ? static_cast<double>(node.sizes[i])
                                          : 1.0 / static_cast<double>(node.children.size()));
  }

  json11::Json::array children;
  for (const WorkspaceNode &child : node.children) children.push_back(node_to_json(child, tab));

  return json11::Json::object{
    {"split", node.orientation == SplitOrientation::Horizontal ? "horizontal" : "vertical"},
    {"sizes", sizes},
    {"children", children},
  };
}

WorkspaceTab tab_from_json(const json11::Json &json, const fs::path &source) {
  WorkspaceTab tab;
  tab.name = json["name"].is_string() && !json["name"].string_value().empty() ? json["name"].string_value() : "tab1";

  const std::vector<json11::Json> panes = json["panes"].array_items();
  tab.panes.reserve(panes.size());
  for (const json11::Json &pane : panes) {
    if (pane.is_object()) tab.panes.push_back(pane_from_json(pane));
  }

  if (!json["root"].is_object()) {
    if (!source.empty()) throw std::runtime_error("Layout tab has no root: " + source.string());
    return make_tab(tab.name);
  }
  tab.root = node_from_json(json["root"], &tab);
  ensure_nonempty_tab(&tab);
  normalize_split_node(&tab.root);
  return tab;
}

json11::Json tab_to_json(const WorkspaceTab &tab) {
  json11::Json::array panes;
  for (const PaneInstance &pane : tab.panes) panes.push_back(pane_to_json(pane));
  return json11::Json::object{
    {"name", tab.name},
    {"panes", panes},
    {"root", node_to_json(tab.root, tab)},
  };
}

void add_default_split(WorkspaceTab *tab, PaneSplit split, PaneInstance pane) {
  if (split_pane(tab, 0, split, std::move(pane))) return;
  tab->panes.push_back(std::move(pane));
}

void draw_dummy_pane(Session &, PaneInstance &) {
}

}  // namespace

static const PaneType kPaneTypes[] = {
  {"empty", "Empty", draw_dummy_pane},
  {"plot", "Plot", draw_plot_pane},
  {"messages", "Messages", draw_messages_pane},
  {"binary", "Binary", draw_binary_pane},
  {"dbc", "DBC", draw_dbc_pane},
  {"signal", "Signal", draw_signal_pane},
  {"historylog", "History", draw_history_log_pane},
  {"find_signal", "Find Signal", draw_find_signal_pane},
  {"find_bits", "Find Bits", draw_find_bits_pane},
  {"browser", "Browser", draw_browser_pane},
  {"logs", "Logs", draw_logs_pane},
  {"map", "Map", draw_map_pane},
  {"computed", "Computed", draw_computed_pane},
  {"camera", "Camera", draw_camera_pane},
};

const PaneType *pane_type(std::string_view id) {
  const auto it = std::find_if(kPaneTypes, kPaneTypes + pane_type_count(), [&](const PaneType &type) {
    return id == type.id;
  });
  return it == kPaneTypes + pane_type_count() ? nullptr : it;
}

const PaneType *pane_types() {
  return kPaneTypes;
}

size_t pane_type_count() {
  return sizeof(kPaneTypes) / sizeof(kPaneTypes[0]);
}

void WorkspaceHistory::reset(const Workspace &workspace) {
  history.clear();
  history.push_back(workspace);
  position = 0;
}

void WorkspaceHistory::push(const Workspace &workspace) {
  if (position < 0) {
    reset(workspace);
    return;
  }
  if (position + 1 < static_cast<int>(history.size())) {
    history.resize(static_cast<size_t>(position + 1));
  }
  history.push_back(workspace);
  if (history.size() > kMaxHistory) history.erase(history.begin());
  position = static_cast<int>(history.size()) - 1;
}

bool workspace_autosave_available(const fs::path &layout_path) {
  return !layout_path.empty();
}

void autosave_workspace(const Workspace &workspace, const fs::path &layout_path, std::string &workspace_status,
                       std::string_view status) {
  if (!workspace_autosave_available(layout_path)) return;
  try {
    save_workspace_draft(workspace, layout_path);
    workspace_status = std::string(status);
  } catch (const std::exception &e) {
    workspace_status = "Workspace autosave failed: " + std::string(e.what());
  }
}

void record_workspace_change(Workspace &workspace, WorkspaceHistory &history, const fs::path &layout_path,
                            std::string &workspace_status, std::string_view status) {
  history.push(workspace);
  autosave_workspace(workspace, layout_path, workspace_status, status);
}

std::optional<int> restore_workspace_snapshot(Workspace &workspace, const Workspace *snapshot, const fs::path &layout_path,
                                             std::string &workspace_status, std::string_view status) {
  if (snapshot == nullptr) return std::nullopt;
  workspace = *snapshot;
  normalize_workspace(&workspace);
  autosave_workspace(workspace, layout_path, workspace_status, status);
  if (!workspace_autosave_available(layout_path)) {
    workspace_status = std::string(status);
  }
  return workspace.current_tab_index;
}

void save_workspace_now(Workspace &workspace, WorkspaceHistory &history, const fs::path &layout_path,
                       std::string &workspace_status) {
  if (!workspace_autosave_available(layout_path)) {
    workspace_status = "No layout file to save";
    return;
  }
  try {
    save_workspace_json(workspace, layout_path);
    clear_workspace_draft(layout_path);
    history.reset(workspace);
    workspace_status = "Saved workspace layout";
  } catch (const std::exception &e) {
    workspace_status = "Workspace save failed: " + std::string(e.what());
  }
}

void clear_workspace_draft_now(const fs::path &layout_path, std::string &workspace_status) {
  if (!workspace_autosave_available(layout_path)) {
    workspace_status = "No layout draft to clear";
    return;
  }
  try {
    clear_workspace_draft(layout_path);
    workspace_status = "Cleared workspace draft";
  } catch (const std::exception &e) {
    workspace_status = "Clear draft failed: " + std::string(e.what());
  }
}

bool WorkspaceHistory::can_undo() const {
  return position > 0;
}

bool WorkspaceHistory::can_redo() const {
  return position >= 0 && position + 1 < static_cast<int>(history.size());
}

const Workspace *WorkspaceHistory::undo() {
  return can_undo() ? &history[static_cast<size_t>(--position)] : nullptr;
}

const Workspace *WorkspaceHistory::redo() {
  return can_redo() ? &history[static_cast<size_t>(++position)] : nullptr;
}

PaneInstance make_pane(std::string type, std::string title, std::string state_json) {
  PaneInstance pane;
  pane.type = std::move(type);
  pane.title = std::move(title);
  pane.state_json = std::move(state_json);
  return pane;
}

WorkspaceTab make_tab(std::string name, PaneInstance pane) {
  WorkspaceTab tab;
  tab.name = name.empty() ? "tab1" : std::move(name);
  tab.panes.push_back(std::move(pane));
  tab.root.is_pane = true;
  tab.root.pane_index = 0;
  return tab;
}

Workspace make_empty_workspace() {
  Workspace workspace;
  workspace.tabs.push_back(make_tab("tab1"));
  return workspace;
}

Workspace make_cabana_workspace() {
  Workspace workspace;
  WorkspaceTab tab = make_tab("Cabana", make_pane("messages", "Messages"));
  add_default_split(&tab, PaneSplit::Right, make_pane("binary", "Binary"));
  add_default_split(&tab, PaneSplit::Bottom, make_pane("signal", "Signal"));
  add_default_split(&tab, PaneSplit::Bottom, make_pane("historylog", "History"));
  workspace.tabs.push_back(std::move(tab));
  workspace.tabs.push_back(make_tab("DBC", make_pane("dbc", "DBC")));
  WorkspaceTab analysis = make_tab("Analysis", make_pane("find_signal", "Find Signal"));
  add_default_split(&analysis, PaneSplit::Right, make_pane("find_bits", "Find Bits"));
  workspace.tabs.push_back(std::move(analysis));
  return workspace;
}

Workspace make_jotpluggler_workspace() {
  Workspace workspace;
  WorkspaceTab tab = make_tab("Jotpluggler", make_pane("plot", "Plot"));
  add_default_split(&tab, PaneSplit::Left, make_pane("browser", "Browser"));
  add_default_split(&tab, PaneSplit::Right, make_pane("map", "Map"));
  add_default_split(&tab, PaneSplit::Bottom, make_pane("logs", "Logs"));
  workspace.tabs.push_back(std::move(tab));
  workspace.tabs.push_back(make_tab("Computed", make_pane("computed", "Computed")));
  return workspace;
}

Workspace make_default_workspace(std::string_view preset) {
  if (preset == "cabana") return make_cabana_workspace();
  if (preset == "jotpluggler") return make_jotpluggler_workspace();
  return make_empty_workspace();
}

WorkspaceTab *active_tab(Workspace *workspace) {
  if (workspace == nullptr || workspace->tabs.empty()) return nullptr;
  const int index = std::clamp(workspace->current_tab_index, 0, static_cast<int>(workspace->tabs.size()) - 1);
  return &workspace->tabs[static_cast<size_t>(index)];
}

const WorkspaceTab *active_tab(const Workspace &workspace) {
  if (workspace.tabs.empty()) return nullptr;
  const int index = std::clamp(workspace.current_tab_index, 0, static_cast<int>(workspace.tabs.size()) - 1);
  return &workspace.tabs[static_cast<size_t>(index)];
}

std::string next_tab_name(const Workspace &workspace, std::string_view base_name) {
  const std::string base(base_name.empty() ? "tab1" : base_name);
  auto exists = [&](std::string_view candidate) {
    return std::any_of(workspace.tabs.begin(), workspace.tabs.end(), [&](const WorkspaceTab &tab) {
      return tab.name == candidate;
    });
  };
  if (!exists(base)) return base;

  int suffix = 2;
  while (exists(base + " " + std::to_string(suffix))) ++suffix;
  return base + " " + std::to_string(suffix);
}

int add_tab(Workspace *workspace, std::string name) {
  if (workspace == nullptr) return -1;
  const std::string tab_name = next_tab_name(*workspace, name.empty() ? "tab1" : name);
  workspace->tabs.push_back(make_tab(tab_name));
  workspace->current_tab_index = static_cast<int>(workspace->tabs.size()) - 1;
  return workspace->current_tab_index;
}

bool duplicate_tab(Workspace *workspace, int tab_index) {
  if (workspace == nullptr || tab_index < 0 || tab_index >= static_cast<int>(workspace->tabs.size())) return false;
  WorkspaceTab tab = workspace->tabs[static_cast<size_t>(tab_index)];
  tab.name = next_tab_name(*workspace, tab.name + " copy");
  workspace->tabs.push_back(std::move(tab));
  workspace->current_tab_index = static_cast<int>(workspace->tabs.size()) - 1;
  return true;
}

bool close_tab(Workspace *workspace, int tab_index) {
  if (workspace == nullptr || tab_index < 0 || tab_index >= static_cast<int>(workspace->tabs.size())) return false;
  if (workspace->tabs.size() == 1) {
    workspace->tabs[0] = make_tab(workspace->tabs[0].name.empty() ? "tab1" : workspace->tabs[0].name);
    workspace->current_tab_index = 0;
    return true;
  }
  workspace->tabs.erase(workspace->tabs.begin() + static_cast<std::ptrdiff_t>(tab_index));
  workspace->current_tab_index = std::clamp(workspace->current_tab_index, 0, static_cast<int>(workspace->tabs.size()) - 1);
  return true;
}

bool rename_tab(Workspace *workspace, int tab_index, std::string name) {
  if (workspace == nullptr || tab_index < 0 || tab_index >= static_cast<int>(workspace->tabs.size()) || name.empty()) return false;
  workspace->tabs[static_cast<size_t>(tab_index)].name = std::move(name);
  return true;
}

int add_pane(WorkspaceTab *tab, PaneInstance pane, std::optional<int> split_target, PaneSplit split) {
  if (tab == nullptr) return -1;
  ensure_nonempty_tab(tab);
  const int target = split_target.value_or(0);
  if (!valid_pane_index(*tab, target)) return -1;
  if (!split_pane(tab, target, split, std::move(pane))) return -1;
  return static_cast<int>(tab->panes.size()) - 1;
}

bool replace_pane(WorkspaceTab *tab, int pane_index, PaneInstance pane) {
  if (tab == nullptr || !valid_pane_index(*tab, pane_index)) return false;
  tab->panes[static_cast<size_t>(pane_index)] = std::move(pane);
  return true;
}

bool split_pane(WorkspaceTab *tab, int pane_index, PaneSplit split, PaneInstance pane) {
  if (tab == nullptr || !valid_pane_index(*tab, pane_index)) return false;
  const int new_pane_index = static_cast<int>(tab->panes.size());
  tab->panes.push_back(std::move(pane));
  if (split_pane_node(&tab->root, pane_index, orientation_for_split(split), split_before(split), new_pane_index)) {
    normalize_split_node(&tab->root);
    return true;
  }
  tab->panes.pop_back();
  return false;
}

bool move_pane(WorkspaceTab *tab, int pane_index, int target_pane_index, PaneSplit split) {
  if (tab == nullptr || !valid_pane_index(*tab, pane_index) || !valid_pane_index(*tab, target_pane_index)) return false;
  if (pane_index == target_pane_index) return false;

  const WorkspaceTab before = *tab;
  PaneInstance moving = tab->panes[static_cast<size_t>(pane_index)];
  if (!remove_pane_from_tree(&tab->root, pane_index)) return false;

  std::vector<PaneInstance> remaining;
  remaining.reserve(tab->panes.size() - 1);
  std::vector<int> mapping(tab->panes.size(), -1);
  for (size_t old_index = 0; old_index < tab->panes.size(); ++old_index) {
    if (static_cast<int>(old_index) == pane_index) continue;
    mapping[old_index] = static_cast<int>(remaining.size());
    remaining.push_back(std::move(tab->panes[old_index]));
  }
  remap_pane_indices(&tab->root, mapping);
  tab->panes = std::move(remaining);
  normalize_split_node(&tab->root);

  const int remapped_target = target_pane_index > pane_index ? target_pane_index - 1 : target_pane_index;
  if (split_pane(tab, remapped_target, split, std::move(moving))) return true;
  *tab = before;
  return false;
}

bool close_pane(WorkspaceTab *tab, int pane_index) {
  if (tab == nullptr || !valid_pane_index(*tab, pane_index)) return false;
  if (tab->panes.size() <= 1) {
    tab->panes[static_cast<size_t>(pane_index)] = make_pane();
    tab->root = WorkspaceNode();
    tab->root.is_pane = true;
    tab->root.pane_index = 0;
    return true;
  }
  if (!remove_pane_from_tree(&tab->root, pane_index)) return false;
  tab->panes.erase(tab->panes.begin() + static_cast<std::ptrdiff_t>(pane_index));
  decrement_pane_indices(&tab->root, pane_index);
  normalize_split_node(&tab->root);
  return true;
}

void normalize_workspace(Workspace *workspace) {
  if (workspace == nullptr) return;
  if (workspace->tabs.empty()) workspace->tabs.push_back(make_tab("tab1"));
  for (WorkspaceTab &tab : workspace->tabs) {
    ensure_nonempty_tab(&tab);
    normalize_split_node(&tab.root);
  }
  workspace->current_tab_index = std::clamp(workspace->current_tab_index, 0, static_cast<int>(workspace->tabs.size()) - 1);
}

std::string workspace_to_json(const Workspace &workspace) {
  Workspace copy = workspace;
  normalize_workspace(&copy);

  json11::Json::array tabs;
  for (const WorkspaceTab &tab : copy.tabs) tabs.push_back(tab_to_json(tab));
  const json11::Json root = json11::Json::object{
    {"version", kVersion},
    {"current_tab_index", copy.current_tab_index},
    {"tabs", tabs},
  };
  return root.dump() + "\n";
}

Workspace workspace_from_json(std::string_view json_text, const fs::path &source) {
  std::string err;
  const json11::Json root = json11::Json::parse(std::string(json_text), err);
  if (!err.empty() || !root.is_object()) {
    throw std::runtime_error(source.empty() ? "Failed to parse workspace JSON"
                                            : "Failed to parse workspace JSON: " + source.string());
  }

  Workspace workspace;
  for (const json11::Json &tab : root["tabs"].array_items()) {
    if (tab.is_object()) workspace.tabs.push_back(tab_from_json(tab, source));
  }
  if (workspace.tabs.empty()) {
    const std::string preset = root["preset"].string_value();
    workspace = make_default_workspace(preset);
  }
  const json11::Json &current = root["current_tab_index"].is_number() ? root["current_tab_index"] : root["currentTabIndex"];
  workspace.current_tab_index = current.is_number() ? current.int_value() : 0;
  normalize_workspace(&workspace);
  return workspace;
}

void save_workspace_json(const Workspace &workspace, const fs::path &path) {
  write_file(path, workspace_to_json(workspace));
}

Workspace load_workspace_json(const fs::path &path) {
  return workspace_from_json(read_file(path), path);
}

fs::path layouts_dir() {
  return fs::path(repo_root()) / "openpilot" / "tools" / "loggy" / "layouts";
}

fs::path autosave_dir() {
  return layouts_dir() / ".autosave";
}

fs::path autosave_path_for_layout(const fs::path &layout_path) {
  const std::string stem = layout_path.empty() ? "untitled" : layout_path.stem().string();
  return autosave_dir() / (sanitize_stem(stem) + ".json");
}

void save_workspace_draft(const Workspace &workspace, const fs::path &layout_path) {
  save_workspace_json(workspace, autosave_path_for_layout(layout_path));
}

WorkspaceLoadResult load_workspace_or_draft(const fs::path &layout_path) {
  const fs::path draft = autosave_path_for_layout(layout_path);
  const bool use_draft = fs::exists(draft);
  return {.workspace = load_workspace_json(use_draft ? draft : layout_path), .loaded_draft = use_draft};
}

void clear_workspace_draft(const fs::path &layout_path) {
  const fs::path draft = autosave_path_for_layout(layout_path);
  if (fs::exists(draft)) fs::remove(draft);
}

}  // namespace loggy
