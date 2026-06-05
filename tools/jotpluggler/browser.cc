#include "tools/jotpluggler/app.h"

#include "imgui_internal.h"

#include <cmath>
#include <cstdio>
#include <unordered_set>

namespace {

constexpr float BROWSER_VALUE_WIDTH = 88.0f;

bool path_matches_filter(const std::string &path, const std::string &lower_filter) {
  if (lower_filter.empty()) return true;
  return lowercase_copy(path).find(lower_filter) != std::string::npos;
}

void insert_browser_path(std::vector<BrowserNode> *nodes, const std::string &path) {
  size_t start = 0;
  while (start < path.size() && path[start] == '/') {
    ++start;
  }
  std::vector<std::string> parts;
  while (start < path.size()) {
    const size_t end = path.find('/', start);
    parts.push_back(path.substr(start, end == std::string::npos ? std::string::npos : end - start));
    if (end == std::string::npos) break;
    start = end + 1;
  }
  if (parts.empty()) {
    return;
  }

  std::vector<BrowserNode> *current_nodes = nodes;
  std::string current_path;
  for (size_t i = 0; i < parts.size(); ++i) {
    if (!current_path.empty()) {
      current_path += "/";
    }
    current_path += parts[i];
    auto it = std::find_if(current_nodes->begin(), current_nodes->end(),
                           [&](const BrowserNode &node) { return node.label == parts[i]; });
    if (it == current_nodes->end()) {
      current_nodes->push_back(BrowserNode{.label = parts[i]});
      it = std::prev(current_nodes->end());
    }
    if (i + 1 == parts.size()) {
      it->full_path = "/" + current_path;
    }
    current_nodes = &it->children;
  }
}

void sort_browser_nodes(std::vector<BrowserNode> *nodes) {
  std::sort(nodes->begin(), nodes->end(), [](const BrowserNode &a, const BrowserNode &b) {
    if (a.children.empty() != b.children.empty()) {
      return !a.children.empty();
    }
    return a.label < b.label;
  });
  for (BrowserNode &node : *nodes) {
    sort_browser_nodes(&node.children);
  }
}

std::vector<BrowserNode> build_browser_tree(const std::vector<std::string> &paths) {
  std::vector<BrowserNode> nodes;
  for (const std::string &path : paths) {
    insert_browser_path(&nodes, path);
  }
  sort_browser_nodes(&nodes);
  return nodes;
}

bool is_deprecated_browser_path(const std::string &path) {
  return path.find("DEPRECATED") != std::string::npos || path.find("/deprecated/") != std::string::npos;
}

std::vector<std::string> visible_browser_paths(const RouteData &route_data, bool show_deprecated_fields) {
  if (show_deprecated_fields) return route_data.paths;
  std::vector<std::string> filtered;
  filtered.reserve(route_data.paths.size());
  for (const std::string &path : route_data.paths) {
    if (!is_deprecated_browser_path(path)) {
      filtered.push_back(path);
    }
  }
  return filtered;
}

bool browser_selection_contains(const UiState &state, std::string_view path) {
  return std::find(state.selected_browser_paths.begin(), state.selected_browser_paths.end(), path)
    != state.selected_browser_paths.end();
}

std::vector<std::string> browser_drag_paths(const UiState &state, const std::string &dragged_path) {
  if (browser_selection_contains(state, dragged_path) && !state.selected_browser_paths.empty()) {
    return state.selected_browser_paths;
  }
  return {dragged_path};
}

std::string encode_browser_drag_payload(const std::vector<std::string> &paths) {
  std::string payload;
  for (size_t i = 0; i < paths.size(); ++i) {
    if (i != 0) {
      payload.push_back('\n');
    }
    payload += paths[i];
  }
  return payload;
}

void set_browser_selection_single(UiState *state, const std::string &path) {
  state->selected_browser_paths = {path};
  state->selected_browser_path = path;
  state->browser_selection_anchor = path;
}

void toggle_browser_selection(UiState *state, const std::string &path) {
  auto it = std::find(state->selected_browser_paths.begin(), state->selected_browser_paths.end(), path);
  if (it == state->selected_browser_paths.end()) {
    state->selected_browser_paths.push_back(path);
  } else {
    state->selected_browser_paths.erase(it);
  }
  state->selected_browser_path = path;
  state->browser_selection_anchor = path;
  if (state->selected_browser_paths.empty()) {
    state->selected_browser_path.clear();
  }
}

void select_browser_range(UiState *state, const std::vector<std::string> &visible_paths, const std::string &clicked_path) {
  if (visible_paths.empty()) {
    set_browser_selection_single(state, clicked_path);
    return;
  }

  const std::string anchor = state->browser_selection_anchor.empty() ? clicked_path : state->browser_selection_anchor;
  const auto anchor_it = std::find(visible_paths.begin(), visible_paths.end(), anchor);
  const auto clicked_it = std::find(visible_paths.begin(), visible_paths.end(), clicked_path);
  if (clicked_it == visible_paths.end()) {
    return;
  }
  if (anchor_it == visible_paths.end()) {
    set_browser_selection_single(state, clicked_path);
    return;
  }

  const auto [begin_it, end_it] = std::minmax(anchor_it, clicked_it);
  std::vector<std::string> selected;
  selected.reserve(static_cast<size_t>(std::distance(begin_it, end_it)) + 1);
  for (auto it = begin_it; it != end_it + 1; ++it) {
    selected.push_back(*it);
  }
  state->selected_browser_paths = std::move(selected);
  state->selected_browser_path = clicked_path;
}

void prune_browser_selection(UiState *state, const std::vector<std::string> &visible_paths) {
  const std::unordered_set<std::string> visible_set(visible_paths.begin(), visible_paths.end());
  auto is_visible = [&](const std::string &path) {
    return visible_set.count(path) > 0;
  };

  state->selected_browser_paths.erase(
    std::remove_if(state->selected_browser_paths.begin(), state->selected_browser_paths.end(),
                   [&](const std::string &path) { return !is_visible(path); }),
    state->selected_browser_paths.end());

  if (!state->selected_browser_path.empty() && !is_visible(state->selected_browser_path)) {
    state->selected_browser_path.clear();
  }
  if (!state->browser_selection_anchor.empty() && !is_visible(state->browser_selection_anchor)) {
    state->browser_selection_anchor.clear();
  }
  if (state->selected_browser_paths.empty()) {
    state->selected_browser_path.clear();
  } else if (state->selected_browser_path.empty()) {
    state->selected_browser_path = state->selected_browser_paths.back();
  }
}

std::optional<double> sample_route_series_value(const RouteSeries &series, double tm, bool stairs) {
  return app_sample_xy_value_at_time(series.times, series.values, stairs, tm);
}

std::string browser_series_value_text(const AppSession &session, const UiState &state, std::string_view path) {
  auto it = session.series_by_path.find(std::string(path));
  if (it == session.series_by_path.end() || it->second == nullptr) return {};

  const RouteSeries &series = *it->second;
  if (series.values.empty()) return {};

  const auto enum_it = session.route_data.enum_info.find(series.path);
  const EnumInfo *enum_info = enum_it == session.route_data.enum_info.end() ? nullptr : &enum_it->second;
  const bool stairs = enum_info != nullptr;

  std::optional<double> value;
  if (state.has_tracker_time) {
    value = sample_route_series_value(series, state.tracker_time, stairs);
  } else {
    value = series.values.back();
  }
  if (!value.has_value()) return {};

  const auto display_it = session.route_data.series_formats.find(series.path);
  const SeriesFormat display_info = display_it == session.route_data.series_formats.end()
    ? compute_series_format(series.values, enum_info != nullptr)
    : display_it->second;

  return format_display_value(*value, display_info, enum_info);
}

bool browser_node_matches(const BrowserNode &node, const std::string &filter) {
  if (filter.empty()) return true;
  if (!node.full_path.empty() && path_matches_filter(node.full_path, filter)) {
    return true;
  }
  for (const BrowserNode &child : node.children) {
    if (browser_node_matches(child, filter)) return true;
  }
  return false;
}

}  // namespace

namespace {

int decimals_needed(double value) {
  const double abs_value = std::abs(value);
  if (abs_value < 1.0e-12) return 0;
  for (int decimals = 0; decimals <= 6; ++decimals) {
    const double scale = std::pow(10.0, decimals);
    if (std::abs(abs_value * scale - std::round(abs_value * scale)) < 1.0e-6) {
      return decimals;
    }
  }
  return 6;
}

void finalize_series_format(SeriesFormat *format) {
  format->digits_before = std::max(format->digits_before, 1);
  format->decimals = std::clamp(format->decimals, 0, 6);
  format->integer_like = format->decimals == 0;
  const int sign_width = format->has_negative ? 1 : 0;
  const int dot_width = format->decimals > 0 ? 1 : 0;
  format->total_width = sign_width + format->digits_before + dot_width + format->decimals;
  std::snprintf(format->fmt, sizeof(format->fmt), "%%%d.%df", format->total_width, format->decimals);
}

}  // namespace

SeriesFormat compute_series_format(const std::vector<double> &values, bool enum_like) {
  SeriesFormat format;
  if (values.empty()) return format;

  const size_t step = std::max<size_t>(1, values.size() / 256);
  bool saw_finite = false;
  bool all_integer = enum_like;
  double min_value = 0.0;
  double max_value = 0.0;
  int max_needed_decimals = 0;

  for (size_t i = 0; i < values.size(); i += step) {
    const double value = values[i];
    if (!std::isfinite(value)) continue;
    if (!saw_finite) {
      min_value = value;
      max_value = value;
      saw_finite = true;
    } else {
      min_value = std::min(min_value, value);
      max_value = std::max(max_value, value);
    }
    if (std::abs(value - std::round(value)) > 1.0e-9) {
      all_integer = false;
    }
    if (!all_integer) {
      max_needed_decimals = std::max(max_needed_decimals, decimals_needed(value));
    }
  }

  if (!saw_finite) return format;

  format.has_negative = min_value < 0.0;
  const double peak = std::max(std::abs(min_value), std::abs(max_value));
  format.digits_before = peak < 1.0 ? 1 : static_cast<int>(std::floor(std::log10(peak))) + 1;

  if (enum_like || all_integer) {
    format.decimals = 0;
  } else if (peak >= 1000.0) {
    format.decimals = std::min(max_needed_decimals, 1);
  } else if (peak >= 100.0) {
    format.decimals = std::min(max_needed_decimals, 2);
  } else {
    format.decimals = std::min(max_needed_decimals, 4);
  }

  finalize_series_format(&format);
  return format;
}

std::string format_display_value(double display_value,
                                 const SeriesFormat &display_info,
                                 const EnumInfo *enum_info) {
  if (!std::isfinite(display_value)) return "---";
  if (enum_info != nullptr) {
    const int idx = static_cast<int>(std::llround(display_value));
    if (idx >= 0 && std::abs(display_value - static_cast<double>(idx)) < 0.01
        && static_cast<size_t>(idx) < enum_info->names.size()
        && !enum_info->names[static_cast<size_t>(idx)].empty()) {
      return enum_info->names[static_cast<size_t>(idx)];
    }
  }
  char buf[64] = {};
  std::snprintf(buf, sizeof(buf), display_info.fmt, display_value);
  return buf;
}

std::vector<std::string> decode_browser_drag_payload(std::string_view payload) {
  std::vector<std::string> out;
  size_t begin = 0;
  while (begin <= payload.size()) {
    const size_t end = payload.find('\n', begin);
    const size_t length = (end == std::string_view::npos ? payload.size() : end) - begin;
    if (length > 0) {
      out.emplace_back(payload.substr(begin, length));
    }
    if (end == std::string_view::npos) break;
    begin = end + 1;
  }
  return out;
}

void collect_visible_leaf_paths(const BrowserNode &node,
                                const std::string &filter,
                                std::vector<std::string> *out) {
  if (!browser_node_matches(node, filter)) {
    return;
  }
  if (node.children.empty()) {
    if (!node.full_path.empty()) {
      out->push_back(node.full_path);
    }
    return;
  }
  for (const BrowserNode &child : node.children) {
    collect_visible_leaf_paths(child, filter, out);
  }
}

void rebuild_browser_nodes(AppSession *session, UiState *state) {
  const std::vector<std::string> paths = visible_browser_paths(session->route_data, state->show_deprecated_fields);
  session->browser_nodes = build_browser_tree(paths);
  prune_browser_selection(state, paths);
}

void rebuild_route_index(AppSession *session) {
  session->series_by_path.clear();
  session->route_data.series_formats.clear();
  for (RouteSeries &series : session->route_data.series) {
    session->series_by_path.emplace(series.path, &series);
    const bool enum_like = session->route_data.enum_info.find(series.path) != session->route_data.enum_info.end();
    session->route_data.series_formats.emplace(series.path, compute_series_format(series.values, enum_like));
  }
}

void draw_browser_node(AppSession *session,
                       const BrowserNode &node,
                       UiState *state,
                       const std::string &filter,
                       const std::vector<std::string> &visible_paths) {
  if (!browser_node_matches(node, filter)) {
    return;
  }

  if (node.children.empty()) {
    const std::string value_text = browser_series_value_text(*session, *state, node.full_path);
    const ImGuiStyle &style = ImGui::GetStyle();
    const ImVec2 row_size(std::max(1.0f, ImGui::GetContentRegionAvail().x), ImGui::GetFrameHeight());
    ImGui::PushID(node.full_path.c_str());
    const bool clicked = ImGui::InvisibleButton("##browser_leaf", row_size);
    const bool hovered = ImGui::IsItemHovered();
    const ImRect rect(ImGui::GetItemRectMin(), ImGui::GetItemRectMax());
    ImDrawList *draw_list = ImGui::GetWindowDrawList();
    if (hovered) {
      const ImU32 bg = ImGui::GetColorU32(ImGuiCol_HeaderHovered);
      draw_list->AddRectFilled(rect.Min, rect.Max, bg, 0.0f);
    }

    const float value_right = rect.Max.x - style.FramePadding.x;
    const float value_left = value_right - (value_text.empty() ? 0.0f : BROWSER_VALUE_WIDTH);
    const float label_left = rect.Min.x + style.FramePadding.x;
    const float label_right = value_text.empty()
      ? rect.Max.x - style.FramePadding.x
      : std::max(label_left + 40.0f, value_left - 10.0f);
    ImGui::RenderTextEllipsis(draw_list,
                              ImVec2(label_left, rect.Min.y + style.FramePadding.y),
                              ImVec2(label_right, rect.Max.y),
                              label_right,
                              node.label.c_str(),
                              nullptr,
                              nullptr);
    if (!value_text.empty()) {
      app_push_mono_font();
      ImGui::PushStyleColor(ImGuiCol_Text, color_rgb(116, 124, 133));
      ImGui::RenderTextClipped(ImVec2(value_left, rect.Min.y + style.FramePadding.y),
                               ImVec2(value_right, rect.Max.y),
                               value_text.c_str(),
                               nullptr,
                               nullptr,
                               ImVec2(1.0f, 0.0f));
      ImGui::PopStyleColor();
      app_pop_mono_font();
    }

    if (clicked) {
      const bool shift_down = ImGui::GetIO().KeyShift;
      const bool ctrl_down = ImGui::GetIO().KeyCtrl || ImGui::GetIO().KeySuper;
      if (shift_down) {
        select_browser_range(state, visible_paths, node.full_path);
      } else if (ctrl_down) {
        toggle_browser_selection(state, node.full_path);
      } else {
        set_browser_selection_single(state, node.full_path);
      }
    }
    if (hovered && ImGui::IsMouseDoubleClicked(0)) {
      set_browser_selection_single(state, node.full_path);
      app_add_curve_to_active_pane(session, state, node.full_path);
    }
    if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID)) {
      const std::vector<std::string> drag_paths = browser_drag_paths(*state, node.full_path);
      const std::string payload = encode_browser_drag_payload(drag_paths);
      ImGui::SetDragDropPayload("JOTP_BROWSER_PATHS", payload.c_str(), payload.size() + 1);
      if (drag_paths.size() == 1) {
        ImGui::TextUnformatted(drag_paths.front().c_str());
      } else {
        ImGui::Text("%zu timeseries", drag_paths.size());
        ImGui::TextUnformatted(drag_paths.front().c_str());
      }
      ImGui::EndDragDropSource();
    }
    ImGui::PopID();
    return;
  }

  ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_SpanAvailWidth;
  const bool open = ImGui::TreeNodeEx(node.label.c_str(), flags);
  if (open) {
    for (const BrowserNode &child : node.children) {
      draw_browser_node(session, child, state, filter, visible_paths);
    }
    ImGui::TreePop();
  }
}
