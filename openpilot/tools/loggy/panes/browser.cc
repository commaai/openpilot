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
#include <functional>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace loggy {
namespace {

// Floor for the value/sparkline sample window anchored at the tracker (see
// enrich_browser_series_row); the user's sparkline_seconds slider can widen it further.
constexpr double kBrowserMinValueLookbackSeconds = 60.0;

struct BrowserState {
  std::string filter;
  // High enough that a full route's series namespace (~11k paths on the demo route) is never
  // silently truncated; the skeleton cache below is what makes this affordable per frame.
  size_t max_rows = 25000;
  int sparkline_seconds = 30;
  bool show_tree = true;
  bool show_deprecated = false;
};

struct BrowserSparkline {
  std::vector<double> values;
  double min = 0.0;
  double max = 0.0;
};

struct BrowserSeriesRow {
  std::string path;
  std::string label;
  bool deprecated = false;
  std::string annotation;
  bool has_value = false;
  std::string value = "--";
  BrowserSparkline sparkline;
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
  if (json["sparkline_seconds"].is_number()) {
    state.sparkline_seconds = std::clamp(json["sparkline_seconds"].int_value(), 1, 120);
  }
  if (json["show_tree"].is_bool()) state.show_tree = json["show_tree"].bool_value();
  if (json["show_deprecated"].is_bool()) state.show_deprecated = json["show_deprecated"].bool_value();
  return state;
}

std::string browser_state_json(const BrowserState &state) {
  return json11::Json(json11::Json::object{
    {"filter", state.filter},
    {"max_rows", static_cast<int>(state.max_rows)},
    {"sparkline_seconds", state.sparkline_seconds},
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
  size_t start_ = 0;
  while (start_ < path.size()) {
    while (start_ < path.size() && path[start_] == '/') ++start_;
    const size_t end = path.find('/', start_);
    const size_t bounded_end = end == std::string_view::npos ? path.size() : end;
    if (bounded_end > start_) out.emplace_back(path.substr(start_, bounded_end - start_));
    if (end == std::string_view::npos) break;
    start_ = end + 1;
  }
  return out;
}

bool browser_filter_focus_requested(bool ctrl_down, bool key_f_pressed, bool want_text_input) {
  return ctrl_down && key_f_pressed && !want_text_input;
}

std::string browser_schema_path(std::string_view path) {
  if (path.empty()) return {};
  const std::vector<std::string> segments = browser_path_segments(path);
  if (segments.empty()) return {};
  std::string out = segments.front();
  for (size_t i = 1; i < segments.size(); ++i) {
    out += " / ";
    out += segments[i];
  }
  return out;
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

BrowserSparkline browser_sparkline_from_view(const SeriesView &view,
                                            size_t max_points = 36,
                                            double window_seconds = 0.0) {
  BrowserSparkline sparkline;
  if (max_points == 0 || view.points.empty()) return sparkline;
  const double min_time = window_seconds > 0.0 ? view.points.back().t - window_seconds
                                               : -std::numeric_limits<double>::infinity();
  std::vector<double> values;
  values.reserve(view.points.size());
  for (const SeriesPoint &point : view.points) {
    if (point.t < min_time || !std::isfinite(point.value)) continue;
    values.push_back(point.value);
  }
  if (values.empty()) return sparkline;

  const size_t step = values.size() <= max_points ? 1 : (values.size() + max_points - 1) / max_points;
  double min_value = std::numeric_limits<double>::infinity();
  double max_value = -std::numeric_limits<double>::infinity();
  sparkline.values.reserve(std::min(values.size(), max_points));
  for (size_t i = 0; i < values.size(); i += step) {
    sparkline.values.push_back(values[i]);
    min_value = std::min(min_value, values[i]);
    max_value = std::max(max_value, values[i]);
  }
  sparkline.min = min_value;
  sparkline.max = max_value;
  return sparkline;
}

// A row's value/sparkline must resample at the current tracker every frame, not freeze at
// whatever the chart's zoom/view range last covered. `sample_range` is anchored at the tracker
// ([tracker - lookback, tracker]), independent of session.view_range.
BrowserSeriesRow enrich_browser_series_row(const Store &store,
                                          BrowserSeriesRow row,
                                          TimeRange sample_range,
                                          double tracker_time,
                                          const BrowserState &state,
                                          size_t max_points = 96) {
  const SeriesView view = store.series(row.path, sample_range.start_, sample_range.end, max_points);
  if (!view.points.empty()) {
    row.has_value = true;
    row.value = browser_format_value(browser_sample_at_time(view.points, tracker_time));
    row.sparkline = browser_sparkline_from_view(view, 36, static_cast<double>(state.sparkline_seconds));
  } else if (sample_range.start_ > 0.0) {
    // Sparse series (carParams and friends publish once) fall outside the tracker-anchored
    // window — sample-hold the newest value from route start instead of showing "--". Only the
    // Value column: an empty sparkline is honest for a signal with nothing in the window.
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
  // Skeleton cache (paths/labels/tree shape only — values and sparklines are sampled per
  // visible row every frame): rebuilding 11k rows plus the grouped tree each frame is what the
  // old 1000-row truncation was hiding.
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

void draw_browser_sparkline(const BrowserSparkline &sparkline) {
  constexpr float width = 92.0f;
  const float height = std::max(18.0f, ImGui::GetTextLineHeight() + 4.0f);
  const ImVec2 pos = ImGui::GetCursorScreenPos();
  ImGui::Dummy(ImVec2(width, height));

  ImDrawList *draw_list = ImGui::GetWindowDrawList();
  const ImVec2 max(pos.x + width, pos.y + height);
  draw_list->AddRectFilled(pos, max, ImGui::GetColorU32(theme().sparkline_bg), 2.0f);
  draw_list->AddRect(pos, max, ImGui::GetColorU32(theme().sparkline_border), 2.0f);
  if (sparkline.values.empty()) {
    draw_list->AddText(ImVec2(pos.x + 4.0f, pos.y + 2.0f), ImGui::GetColorU32(ImGuiCol_TextDisabled), "--");
    return;
  }

  const double raw_span = sparkline.max - sparkline.min;
  const double span = std::max(raw_span, 1e-9);
  std::vector<ImVec2> points;
  points.reserve(sparkline.values.size());
  for (size_t i = 0; i < sparkline.values.size(); ++i) {
    const float x = pos.x + 2.0f + (width - 4.0f) * (sparkline.values.size() == 1 ? 0.5f : static_cast<float>(i) / static_cast<float>(sparkline.values.size() - 1));
    const double normalized = raw_span <= 1e-9 ? 0.5 : (sparkline.max - sparkline.values[i]) / span;
    const float y = pos.y + 2.0f + (height - 4.0f) * static_cast<float>(normalized);
    points.push_back(ImVec2(x, y));
  }
  if (points.size() == 1) {
    draw_list->AddCircleFilled(points.front(), 2.0f, ImGui::GetColorU32(theme().sparkline_line));
  } else {
    draw_list->AddPolyline(points.data(), static_cast<int>(points.size()), ImGui::GetColorU32(theme().sparkline_line), 0, 1.5f);
  }
}

void draw_browser_row(Session &session, const BrowserSeriesRow &row) {
  ImGui::TableNextRow(ImGuiTableRowFlags_None, std::max(ImGui::GetFrameHeight(), ImGui::GetTextLineHeight() + 8.0f));
  ImGui::PushID(row.path.c_str());

  ImGui::TableSetColumnIndex(0);
  const bool selected = false;
  std::string label = row.label;
  if (!row.annotation.empty()) {
    label += "  [" + row.annotation + "]";
  }
  ImGui::Selectable(label.c_str(), selected, ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowOverlap);
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

  ImGui::TableSetColumnIndex(1);
  push_mono_font();
  ImGui::TextUnformatted(row.value.c_str());
  pop_mono_font();

  ImGui::TableSetColumnIndex(2);
  draw_browser_sparkline(row.sparkline);

  ImGui::TableSetColumnIndex(3);
  push_mono_font();
  ImGui::TextUnformatted(browser_schema_path(row.path).c_str());
  pop_mono_font();

  ImGui::PopID();
}

void draw_browser_tree_node(Session &session, const BrowserTreeNode &node,
                            const Store &store,
                            TimeRange sample_range,
                            double tracker_time,
                            const BrowserState &state) {
  if (node.leaf) {
    draw_browser_row(session, enrich_browser_series_row(store, node.series, sample_range, tracker_time, state));
    return;
  }

  if (node.path != "/") {
    ImGui::TableNextRow(ImGuiTableRowFlags_None, std::max(ImGui::GetFrameHeight(), ImGui::GetTextLineHeight() + 8.0f));
    ImGui::TableSetColumnIndex(0);
    ImGui::PushID(node.path.c_str());
    const bool open = ImGui::TreeNodeEx("##browser_group",
                                        ImGuiTreeNodeFlags_SpanAvailWidth,
                                        "%s (%zu)", node.label.c_str(), node.visible_leaf_count);
    ImGui::TableSetColumnIndex(3);
    push_mono_font();
    ImGui::TextUnformatted(browser_schema_path(node.path).c_str());
    pop_mono_font();
    if (open) {
      for (const BrowserTreeNode &child : node.children) {
        draw_browser_tree_node(session, child, store, sample_range, tracker_time, state);
      }
      ImGui::TreePop();
    }
    ImGui::PopID();
    return;
  }

  for (const BrowserTreeNode &child : node.children) {
    draw_browser_tree_node(session, child, store, sample_range, tracker_time, state);
  }
}

}  // namespace

void draw_browser_pane(Session &session, PaneInstance &pane) {
  BrowserPaneTransientState &transient = browser_pane_transient_state(pane);
  BrowserState &state = browser_pane_state(pane);
  bool changed = false;

  std::array<char, 160> filter_buf{};
  std::snprintf(filter_buf.data(), filter_buf.size(), "%s", state.filter.c_str());
  const ImGuiIO &io = ImGui::GetIO();
  const bool focus_filter = browser_filter_focus_requested(io.KeyCtrl, ImGui::IsKeyPressed(ImGuiKey_F, false),
                                                          io.WantTextInput);
  const float filter_width = std::clamp(ImGui::GetContentRegionAvail().x * 0.50f, 140.0f, 300.0f);
  ImGui::SetNextItemWidth(filter_width);
  if (focus_filter) ImGui::SetKeyboardFocusHere();
  if (ImGui::InputTextWithHint("##browser_search", "Search paths", filter_buf.data(), filter_buf.size())) {
    state.filter = filter_buf.data();
    changed = true;
  }
  // Cumulative thresholds: once one guard fails, every later one fails too and wraps its own row.
  constexpr float kSparkWidth = 150.0f, kTreeWidth = 80.0f, kDeprecatedWidth = 120.0f;
  if (ImGui::GetContentRegionAvail().x > filter_width + kSparkWidth) ImGui::SameLine();
  int sparkline_seconds = state.sparkline_seconds;
  ImGui::SetNextItemWidth(96.0f);
  if (ImGui::SliderInt("Spark", &sparkline_seconds, 1, 120, "%ds", ImGuiSliderFlags_AlwaysClamp)) {
    state.sparkline_seconds = sparkline_seconds;
    changed = true;
  }
  if (ImGui::GetContentRegionAvail().x > filter_width + kSparkWidth + kTreeWidth) ImGui::SameLine();
  changed = ImGui::Checkbox("Tree", &state.show_tree) || changed;
  if (ImGui::GetContentRegionAvail().x > filter_width + kSparkWidth + kTreeWidth + kDeprecatedWidth) ImGui::SameLine();
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

  if (visible_count == 0) {
    ImGui::TextDisabled("No series in store or filter");
    return;
  }

  constexpr ImGuiTableFlags flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV |
                                    ImGuiTableFlags_BordersOuter | ImGuiTableFlags_SizingStretchProp |
                                    ImGuiTableFlags_ScrollY;
  if (!ImGui::BeginTable("##loggy_browser_series", 4, flags, ImGui::GetContentRegionAvail())) return;
  ImGui::TableSetupColumn("Field", ImGuiTableColumnFlags_WidthFixed, state.show_tree ? 190.0f : 126.0f);
  ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthFixed, 88.0f);
  ImGui::TableSetupColumn("Spark", ImGuiTableColumnFlags_WidthFixed, 102.0f);
  ImGui::TableSetupColumn("Path", ImGuiTableColumnFlags_WidthStretch);
  ImGui::TableHeadersRow();

  // Sample anchored at the tracker, not session.view_range (the chart's zoom window) -- rows
  // must keep ticking during playback/seeks regardless of what any plot pane has zoomed to. The
  // lookback is generous versus the sparkline window so a value doesn't drop to "--" just
  // because a slow-updating signal is older than the visible spark.
  const double tracker_time = session.playback.tracker_time();
  const double lookback = std::max(kBrowserMinValueLookbackSeconds, static_cast<double>(state.sparkline_seconds));
  const TimeRange sample_range{tracker_time - lookback, tracker_time};
  if (state.show_tree) {
    draw_browser_tree_node(session, tree, session.store, sample_range, tracker_time, state);
    ImGui::EndTable();
    return;
  }

  ImGuiListClipper clipper;
  clipper.Begin(static_cast<int>(rows.size()), std::max(ImGui::GetFrameHeight(), ImGui::GetTextLineHeight() + 8.0f));
  while (clipper.Step()) {
    for (int row_idx = clipper.DisplayStart; row_idx < clipper.DisplayEnd; ++row_idx) {
      draw_browser_row(session, enrich_browser_series_row(session.store, rows[static_cast<size_t>(row_idx)],
                                                          sample_range, tracker_time, state));
    }
  }
  ImGui::EndTable();
}

}  // namespace loggy
