#include "tools/loggy/panes/browser.h"

#include "tools/loggy/backend/session.h"
#include "tools/loggy/backend/store.h"
#include "tools/loggy/shell/theme.h"
#include "tools/loggy/shell/workspace.h"

#include "imgui.h"
#include "json11/json11.hpp"

#include <algorithm>
#include <array>
#include <any>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace loggy {
namespace {

// A value shown against the tracker samples a trailing window; a slow-updating signal keeps its
// last value rather than dropping to "--" the moment it's older than this.
constexpr double kBrowserValueLookbackSeconds = 60.0;

// Special draggable sources at the top of the browser (jotpluggler parity): drop one on a pane
// to turn it into a Map or Camera. camera_view names match camera_view_from_layout_name().
struct SpecialItem {
  const char *id;
  const char *label;
  const char *pane_type;
  const char *state_json;
};
constexpr std::array<SpecialItem, 5> kSpecialItems = {{
  {"map", "Map", "map", "{}"},
  {"camera_road", "Road Camera", "camera", R"({"camera_view":"road"})"},
  {"camera_driver", "Driver Camera", "camera", R"({"camera_view":"driver"})"},
  {"camera_wide_road", "Wide Road Camera", "camera", R"({"camera_view":"wide_road"})"},
  {"camera_qroad", "qRoad Camera", "camera", R"({"camera_view":"qroad"})"},
}};

struct BrowserState {
  std::string filter;
  // High enough that a full route's series namespace (~11k paths on the demo route) is never
  // silently truncated; the skeleton cache below is what makes this affordable per frame.
  size_t max_rows = 25000;
  bool show_tree = true;
  bool show_deprecated = false;
};

struct BrowserSeriesRow {
  std::string path;
  std::string label;
  bool deprecated = false;
  std::string annotation;
  bool has_value = false;
  std::string value = "--";
};

struct BrowserTreeNode {
  std::string label;
  std::string path;
  bool leaf = false;
  bool deprecated = false;
  size_t visible_leaf_count = 0;
  BrowserSeriesRow series;
  std::vector<BrowserTreeNode> children;
};

BrowserState parse_browser_state(std::string_view state_json) {
  BrowserState state;
  std::string err;
  const json11::Json json = json11::Json::parse(std::string(state_json), err);
  if (!err.empty() || !json.is_object()) return state;
  if (json["filter"].is_string()) state.filter = json["filter"].string_value();
  if (json["max_rows"].is_number()) {
    state.max_rows = static_cast<size_t>(std::clamp(json["max_rows"].int_value(), 1, 50000));
  }
  if (json["show_tree"].is_bool()) state.show_tree = json["show_tree"].bool_value();
  if (json["show_deprecated"].is_bool()) state.show_deprecated = json["show_deprecated"].bool_value();
  return state;
}

std::string browser_state_json(const BrowserState &state) {
  return json11::Json(json11::Json::object{
    {"filter", state.filter},
    {"max_rows", static_cast<int>(state.max_rows)},
    {"show_tree", state.show_tree},
    {"show_deprecated", state.show_deprecated},
  }).dump();
}

std::string browser_leaf_label(std::string_view path) {
  if (path.empty()) return "series";
  const size_t slash = path.find_last_of('/');
  const std::string_view label = slash == std::string_view::npos ? path : path.substr(slash + 1);
  return label.empty() ? std::string(path) : std::string(label);
}

std::string browser_lower_text(std::string_view text) {
  std::string out(text);
  std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return out;
}

bool browser_path_matches_filter(std::string_view path, std::string_view filter) {
  if (filter.empty()) return true;
  const std::string haystack = browser_lower_text(path);
  const std::string needle = browser_lower_text(filter);
  return haystack.find(needle) != std::string::npos;
}

bool browser_path_is_deprecated(std::string_view path) {
  return path.find("DEPRECATED") != std::string_view::npos ||
         path.find("/deprecated/") != std::string_view::npos;
}

std::string browser_metadata_annotation(bool deprecated) {
  return deprecated ? "DEPRECATED" : std::string();
}

std::vector<BrowserSeriesRow> prepare_browser_series_rows(const Store &store, const BrowserState &state) {
  const std::vector<std::string> paths = store.series_paths();
  std::vector<BrowserSeriesRow> rows;
  rows.reserve(std::min(paths.size(), state.max_rows));
  for (const std::string &path : paths) {
    const bool deprecated = browser_path_is_deprecated(path);
    if (deprecated && !state.show_deprecated) continue;
    if (!browser_path_matches_filter(path, state.filter)) continue;
    rows.push_back({
      .path = path,
      .label = browser_leaf_label(path),
      .deprecated = deprecated,
      .annotation = browser_metadata_annotation(deprecated),
    });
    if (rows.size() >= state.max_rows) break;
  }
  return rows;
}

std::vector<std::string> browser_path_segments(std::string_view path) {
  std::vector<std::string> out;
  size_t start = 0;
  while (start < path.size()) {
    while (start < path.size() && path[start] == '/') ++start;
    const size_t end = path.find('/', start);
    const size_t bounded_end = end == std::string_view::npos ? path.size() : end;
    if (bounded_end > start) out.emplace_back(path.substr(start, bounded_end - start));
    if (end == std::string_view::npos) break;
    start = end + 1;
  }
  return out;
}

bool browser_filter_focus_requested(bool ctrl_down, bool key_f_pressed, bool want_text_input) {
  return ctrl_down && key_f_pressed && !want_text_input;
}

BrowserTreeNode *browser_find_or_add_group(BrowserTreeNode *parent,
                                                  const std::string &label,
                                                  const std::string &path) {
  for (BrowserTreeNode &child : parent->children) {
    if (!child.leaf && child.label == label) return &child;
  }
  parent->children.push_back(BrowserTreeNode{
    .label = label,
    .path = path,
  });
  return &parent->children.back();
}

size_t browser_count_visible_leaves(BrowserTreeNode *node) {
  if (node == nullptr) return 0;
  if (node->leaf) {
    node->visible_leaf_count = 1;
    return 1;
  }
  size_t count = 0;
  for (BrowserTreeNode &child : node->children) count += browser_count_visible_leaves(&child);
  node->visible_leaf_count = count;
  return count;
}

BrowserTreeNode prepare_browser_tree(const Store &store, const BrowserState &state) {
  BrowserTreeNode root;
  root.label = "Series";
  root.path = "/";

  const std::vector<BrowserSeriesRow> rows = prepare_browser_series_rows(store, state);
  for (const BrowserSeriesRow &row : rows) {
    const std::vector<std::string> segments = browser_path_segments(row.path);
    BrowserTreeNode *parent = &root;
    std::string prefix;
    for (size_t i = 0; i < segments.size(); ++i) {
      prefix += "/" + segments[i];
      const bool leaf = i + 1 == segments.size();
      if (leaf) {
        parent->children.push_back(BrowserTreeNode{
          .label = segments[i],
          .path = row.path,
          .leaf = true,
          .deprecated = row.deprecated,
          .visible_leaf_count = 1,
          .series = row,
        });
      } else {
        parent = browser_find_or_add_group(parent, segments[i], prefix);
      }
    }
  }

  browser_count_visible_leaves(&root);
  return root;
}

double browser_sample_at_time(const std::vector<SeriesPoint> &points, double time) {
  if (points.empty()) return 0.0;
  if (time <= points.front().t) return points.front().value;
  if (time >= points.back().t) return points.back().value;

  const auto upper = std::lower_bound(points.begin(), points.end(), time, [](const SeriesPoint &point, double value) {
    return point.t < value;
  });
  if (upper == points.begin()) return points.front().value;
  if (upper == points.end()) return points.back().value;
  const SeriesPoint &hi = *upper;
  const SeriesPoint &lo = *(upper - 1);
  if (hi.t <= lo.t) return lo.value;
  const double alpha = (time - lo.t) / (hi.t - lo.t);
  return lo.value + (hi.value - lo.value) * alpha;
}

std::string browser_format_value(double value) {
  if (!std::isfinite(value)) return "--";
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.6g", value);
  return buf;
}

// A row's value must resample at the current tracker every frame, not freeze at whatever the
// chart's zoom/view range last covered. `sample_range` is anchored at the tracker
// ([tracker - lookback, tracker]), independent of session.view_range.
BrowserSeriesRow enrich_browser_series_row(const Store &store,
                                          BrowserSeriesRow row,
                                          TimeRange sample_range,
                                          double tracker_time,
                                          size_t max_points = 96) {
  const SeriesView view = store.series(row.path, sample_range.start_, sample_range.end, max_points);
  if (!view.points.empty()) {
    row.has_value = true;
    row.value = browser_format_value(browser_sample_at_time(view.points, tracker_time));
  } else if (sample_range.start_ > 0.0) {
    // Sparse series (carParams and friends publish once) fall outside the tracker-anchored
    // window — sample-hold the newest value from route start instead of showing "--".
    const SeriesView held = store.series(row.path, 0.0, sample_range.end, 2);
    if (!held.points.empty()) {
      row.has_value = true;
      row.value = browser_format_value(held.points.back().value);
    }
  }
  return row;
}

struct BrowserPaneTransientState {
  BrowserState state;
  std::string state_json;
  bool valid = false;
  // Skeleton cache (paths/labels/tree shape only — values are sampled per visible row every
  // frame): rebuilding 11k rows plus the grouped tree each frame is what the old 1000-row
  // truncation was hiding.
  uint64_t skeleton_generation = std::numeric_limits<uint64_t>::max();
  std::string skeleton_key;
  double skeleton_built_at = -1.0e9;
  BrowserTreeNode skeleton_tree;
  std::vector<BrowserSeriesRow> skeleton_rows;
};

BrowserPaneTransientState &browser_pane_transient_state(PaneInstance &pane) {
  if (BrowserPaneTransientState *state = std::any_cast<BrowserPaneTransientState>(&pane.transient_state)) {
    return *state;
  }
  pane.transient_state = BrowserPaneTransientState{};
  return std::any_cast<BrowserPaneTransientState &>(pane.transient_state);
}

BrowserState &browser_pane_state(PaneInstance &pane) {
  BrowserPaneTransientState &transient = browser_pane_transient_state(pane);
  if (!transient.valid || transient.state_json != pane.state_json) {
    transient.state = parse_browser_state(pane.state_json);
    transient.state_json = pane.state_json;
    transient.valid = true;
  }
  return transient.state;
}

// jotpluggler's dense list row: label on the left, mono value right-aligned. `value_col` is the
// x of the value column's left edge (window-relative); empty when the row is a group header.
void draw_browser_value_cell(const char *value) {
  if (value == nullptr || value[0] == '\0') return;
  const float col_w = 84.0f;
  const float avail = ImGui::GetContentRegionAvail().x;
  ImGui::SameLine(0.0f, 0.0f);
  push_mono_font();
  const ImVec2 text_size = ImGui::CalcTextSize(value);
  const float offset = std::max(0.0f, avail - std::min(col_w, text_size.x));
  ImGui::SameLine(0.0f, offset);
  ImGui::PushStyleColor(ImGuiCol_Text, theme().text_muted);
  ImGui::TextUnformatted(value);
  ImGui::PopStyleColor();
  pop_mono_font();
}

void draw_browser_special_items(Session &session) {
  for (const SpecialItem &item : kSpecialItems) {
    ImGui::PushID(item.id);
    ImGui::Selectable(item.label, false, ImGuiSelectableFlags_AllowOverlap);
    if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID)) {
      ImGui::SetDragDropPayload(kLoggySpecialItemPayload, item.id, std::strlen(item.id) + 1);
      ImGui::TextUnformatted(item.label);
      ImGui::EndDragDropSource();
    }
    ImGui::PopID();
  }
}

void draw_browser_row(Session &session, const BrowserSeriesRow &row) {
  ImGui::PushID(row.path.c_str());
  std::string label = row.label;
  if (!row.annotation.empty()) label += "  [" + row.annotation + "]";
  ImGui::Selectable(label.c_str(), false, ImGuiSelectableFlags_AllowOverlap);
  // Double-click adds the series to the tab's plot — jotpluggler's fastest add path.
  if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
    session.pending_plot_series = row.path;
    session.pending_plot_series_age = 0;
  }
  if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID)) {
    ImGui::SetDragDropPayload(kLoggySeriesPathPayload, row.path.c_str(), row.path.size() + 1);
    ImGui::TextUnformatted(row.path.c_str());
    ImGui::EndDragDropSource();
  }
  draw_browser_value_cell(row.has_value ? row.value.c_str() : nullptr);
  ImGui::PopID();
}

void draw_browser_tree_node(Session &session, const BrowserTreeNode &node,
                            const Store &store,
                            TimeRange sample_range,
                            double tracker_time) {
  if (node.leaf) {
    draw_browser_row(session, enrich_browser_series_row(store, node.series, sample_range, tracker_time));
    return;
  }

  if (node.path != "/") {
    ImGui::PushID(node.path.c_str());
    ImGui::SetNextItemWidth(-1.0f);
    const bool open = ImGui::TreeNodeEx("##browser_group", ImGuiTreeNodeFlags_SpanAvailWidth,
                                        "%s (%zu)", node.label.c_str(), node.visible_leaf_count);
    if (open) {
      for (const BrowserTreeNode &child : node.children) {
        draw_browser_tree_node(session, child, store, sample_range, tracker_time);
      }
      ImGui::TreePop();
    }
    ImGui::PopID();
    return;
  }

  for (const BrowserTreeNode &child : node.children) {
    draw_browser_tree_node(session, child, store, sample_range, tracker_time);
  }
}

}  // namespace

SpecialItemPane browser_special_item_pane(std::string_view id) {
  for (const SpecialItem &item : kSpecialItems) {
    if (id == item.id) return {item.pane_type, item.label, item.state_json};
  }
  return {};
}

void draw_browser_pane(Session &session, PaneInstance &pane) {
  BrowserPaneTransientState &transient = browser_pane_transient_state(pane);
  BrowserState &state = browser_pane_state(pane);
  bool changed = false;

  // Layout selector (jotpluggler parity): pick a saved layout to load it — also how you recover
  // a tab you closed. Deferred to the shell, which reloads the whole workspace next frame.
  const std::string current_layout = session.workspace_layout_path.empty()
                                    ? std::string("untitled")
                                    : session.workspace_layout_path.stem().string();
  ImGui::SetNextItemWidth(-1.0f);
  if (ImGui::BeginCombo("##layout", current_layout.c_str())) {
    for (const std::string &name : available_layout_names()) {
      if (ImGui::Selectable(name.c_str(), name == current_layout)) {
        session.pending_layout_load = layouts_dir() / (name + ".json");
      }
    }
    ImGui::EndCombo();
  }

  std::array<char, 160> filter_buf{};
  std::snprintf(filter_buf.data(), filter_buf.size(), "%s", state.filter.c_str());
  const ImGuiIO &io = ImGui::GetIO();
  const bool focus_filter = browser_filter_focus_requested(io.KeyCtrl, ImGui::IsKeyPressed(ImGuiKey_F, false),
                                                          io.WantTextInput);
  ImGui::SetNextItemWidth(-1.0f);
  if (focus_filter) ImGui::SetKeyboardFocusHere();
  if (ImGui::InputTextWithHint("##browser_search", "Search", filter_buf.data(), filter_buf.size())) {
    state.filter = filter_buf.data();
    changed = true;
  }
  changed = ImGui::Checkbox("Tree", &state.show_tree) || changed;
  ImGui::SameLine();
  changed = ImGui::Checkbox("Deprecated", &state.show_deprecated) || changed;
  if (changed) {
    pane.state_json = browser_state_json(state);
    transient.state_json = pane.state_json;
  }

  const size_t total = session.store.series_path_count();
  char skeleton_key[128];
  std::snprintf(skeleton_key, sizeof(skeleton_key), "%d|%d|%zu|%zu", state.show_tree ? 1 : 0,
                state.show_deprecated ? 1 : 0, state.max_rows, std::hash<std::string>{}(state.filter));
  // During route load the store generation bumps every frame (frame-budgeted drain), which
  // would rebuild this 11k-row skeleton per frame; new DATA only needs a 4 Hz refresh. A KEY
  // change (filter/toggles) rebuilds immediately — that's direct user input.
  const bool key_changed = transient.skeleton_key != skeleton_key;
  const bool data_changed = transient.skeleton_generation != session.store.generation();
  if (key_changed || (data_changed && ImGui::GetTime() - transient.skeleton_built_at > 0.25)) {
    transient.skeleton_tree = state.show_tree ? prepare_browser_tree(session.store, state) : BrowserTreeNode{};
    transient.skeleton_rows = state.show_tree ? std::vector<BrowserSeriesRow>{}
                                              : prepare_browser_series_rows(session.store, state);
    transient.skeleton_generation = session.store.generation();
    transient.skeleton_key = skeleton_key;
    transient.skeleton_built_at = ImGui::GetTime();
  }
  const BrowserTreeNode &tree = transient.skeleton_tree;
  const std::vector<BrowserSeriesRow> &rows = transient.skeleton_rows;
  const size_t visible_count = state.show_tree ? tree.visible_leaf_count : rows.size();
  ImGui::TextDisabled("%zu/%zu series", visible_count, total);

  // Dense scrolling list (jotpluggler): tight rows, no table chrome, no sparkline column.
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8.0f, 3.0f));
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(6.0f, 2.0f));
  if (ImGui::BeginChild("##browser_list", ImGui::GetContentRegionAvail(), false)) {
    // Special sources first, then a separator, then the series tree — jotpluggler's order.
    draw_browser_special_items(session);
    ImGui::Dummy(ImVec2(0.0f, 2.0f));
    ImGui::Separator();
    ImGui::Dummy(ImVec2(0.0f, 2.0f));

    const double tracker_time = session.playback.tracker_time();
    const TimeRange sample_range{tracker_time - kBrowserValueLookbackSeconds, tracker_time};
    if (visible_count == 0) {
      ImGui::TextDisabled("No series in store or filter");
    } else if (state.show_tree) {
      draw_browser_tree_node(session, tree, session.store, sample_range, tracker_time);
    } else {
      ImGuiListClipper clipper;
      clipper.Begin(static_cast<int>(rows.size()), ImGui::GetFrameHeight());
      while (clipper.Step()) {
        for (int row_idx = clipper.DisplayStart; row_idx < clipper.DisplayEnd; ++row_idx) {
          draw_browser_row(session, enrich_browser_series_row(session.store, rows[static_cast<size_t>(row_idx)],
                                                              sample_range, tracker_time));
        }
      }
    }
  }
  ImGui::EndChild();
  ImGui::PopStyleVar(2);
}

}  // namespace loggy
