#include "tools/loggy/panes/plot.h"

#include "tools/loggy/backend/session.h"
#include "tools/loggy/panes/browser.h"
#include "tools/loggy/shell/theme.h"
#include "tools/loggy/shell/workspace.h"

#include "json11/json11.hpp"

#include "imgui.h"
#include "implot.h"

#include <algorithm>
#include <any>
#include <array>
#include <cstdio>
#include <cmath>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace loggy {
namespace {

enum class PlotSeriesStyle {
  Auto,
  Line,
  Step,
  Scatter,
};

enum class PlotSeriesTransform {
  None,
  Derivative,
  Scale,
};

struct PlotYLimits {
  bool min_enabled = false;
  bool max_enabled = false;
  double min = 0.0;
  double max = 0.0;

  bool active() const { return min_enabled || max_enabled; }
};

struct PlotSeriesRequest {
  std::string path;
  std::string label;
  std::string color;
  PlotSeriesTransform transform = PlotSeriesTransform::None;
  double derivative_dt = 0.0;
  double scale = 1.0;
  double offset = 0.0;
  bool force_stairs = false;
  json11::Json custom_python;
};

struct PreparedPlotSeries {
  PlotSeriesRequest request;
  SeriesView view;
  std::vector<double> xs;
  std::vector<double> ys;
  bool stairs = false;
  bool has_tracker_value = false;
  double tracker_value = 0.0;
};

struct PlotYAxisBounds {
  bool active = false;
  double min = 0.0;
  double max = 1.0;
};

struct PlotZoomRange {
  double start_ = 0.0;
  double end = 0.0;
};

std::string plot_label_from_path(std::string_view path) {
  if (path.empty()) return "series";
  const size_t slash = path.find_last_of('/');
  std::string label = slash == std::string_view::npos ? std::string(path) : std::string(path.substr(slash + 1));
  return label.empty() ? std::string(path) : label;
}

PlotSeriesStyle plot_series_style_from_string(std::string_view style) {
  if (style == "line") return PlotSeriesStyle::Line;
  if (style == "step" || style == "stairs") return PlotSeriesStyle::Step;
  if (style == "scatter") return PlotSeriesStyle::Scatter;
  return PlotSeriesStyle::Auto;
}

const char *plot_series_style_token(PlotSeriesStyle style) {
  switch (style) {
    case PlotSeriesStyle::Line: return "line";
    case PlotSeriesStyle::Step: return "step";
    case PlotSeriesStyle::Scatter: return "scatter";
    case PlotSeriesStyle::Auto:
    default: return "auto";
  }
}

PlotSeriesStyle plot_series_style_node(const json11::Json &state) {
  if (!state.is_object()) return PlotSeriesStyle::Auto;
  if (state["style"].is_string()) return plot_series_style_from_string(state["style"].string_value());
  if (state["series_type"].is_string()) return plot_series_style_from_string(state["series_type"].string_value());
  if (state["stairs"].bool_value() || state["step"].bool_value()) return PlotSeriesStyle::Step;
  return PlotSeriesStyle::Auto;
}

PlotSeriesStyle parse_plot_series_style(std::string_view state_json) {
  std::string err;
  const json11::Json state = json11::Json::parse(std::string(state_json), err);
  if (!err.empty()) return PlotSeriesStyle::Auto;
  return plot_series_style_node(state);
}

PlotYLimits parse_plot_y_limits_node(const json11::Json &state) {
  PlotYLimits limits;
  if (!state.is_object()) return limits;
  const json11::Json &node = state["y_limits"].is_object() ? state["y_limits"] : state;
  if (!node.is_object()) return limits;

  const bool min_flag = !node["min_enabled"].is_bool() || node["min_enabled"].bool_value();
  const bool max_flag = !node["max_enabled"].is_bool() || node["max_enabled"].bool_value();
  if (node["min"].is_number() && min_flag && std::isfinite(node["min"].number_value())) {
    limits.min_enabled = true;
    limits.min = node["min"].number_value();
  }
  if (node["max"].is_number() && max_flag && std::isfinite(node["max"].number_value())) {
    limits.max_enabled = true;
    limits.max = node["max"].number_value();
  }
  return limits;
}

PlotYLimits parse_plot_y_limits(std::string_view state_json) {
  std::string err;
  const json11::Json state = json11::Json::parse(std::string(state_json), err);
  if (!err.empty()) return {};
  return parse_plot_y_limits_node(state);
}

json11::Json plot_y_limits_json(const PlotYLimits &limits) {
  json11::Json::object out;
  if (limits.min_enabled && std::isfinite(limits.min)) out["min"] = limits.min;
  if (limits.max_enabled && std::isfinite(limits.max)) out["max"] = limits.max;
  return out;
}

bool plot_zoom_range_valid(const PlotZoomRange &range) {
  return std::isfinite(range.start_) && std::isfinite(range.end) && range.end > range.start_;
}

std::vector<PlotZoomRange> parse_plot_zoom_history_node(const json11::Json &state) {
  std::vector<PlotZoomRange> history;
  if (!state.is_object() || !state["x_zoom_history"].is_array()) return history;
  for (const json11::Json &item : state["x_zoom_history"].array_items()) {
    if (!item.is_object() || !item["start_"].is_number() || !item["end"].is_number()) continue;
    PlotZoomRange range{.start_ = item["start_"].number_value(), .end = item["end"].number_value()};
    if (plot_zoom_range_valid(range)) history.push_back(range);
  }
  return history;
}

std::vector<PlotZoomRange> parse_plot_zoom_history(std::string_view state_json) {
  std::string err;
  const json11::Json state = json11::Json::parse(std::string(state_json), err);
  if (!err.empty()) return {};
  return parse_plot_zoom_history_node(state);
}

json11::Json plot_zoom_history_json(const std::vector<PlotZoomRange> &history) {
  json11::Json::array out;
  out.reserve(history.size());
  for (const PlotZoomRange &range : history) {
    if (!plot_zoom_range_valid(range)) continue;
    out.push_back(json11::Json::object{{"start_", range.start_}, {"end", range.end}});
  }
  return out;
}

bool plot_integer_like(const std::vector<double> &ys) {
  if (ys.empty()) return false;
  for (double y : ys) {
    if (!std::isfinite(y)) continue;
    if (std::abs(y - std::round(y)) > 1.0e-9) return false;
  }
  return true;
}

double plot_sample_at_time(const std::vector<double> &xs, const std::vector<double> &ys, bool stairs, double time) {
  if (xs.empty() || xs.size() != ys.size()) return 0.0;
  if (time <= xs.front()) return ys.front();
  if (time >= xs.back()) return ys.back();

  const auto upper = std::lower_bound(xs.begin(), xs.end(), time);
  if (upper == xs.begin()) return ys.front();
  if (upper == xs.end()) return ys.back();

  const size_t hi = static_cast<size_t>(std::distance(xs.begin(), upper));
  const size_t lo = hi - 1;
  if (stairs || xs[hi] <= xs[lo]) return ys[lo];
  const double alpha = (time - xs[lo]) / (xs[hi] - xs[lo]);
  return ys[lo] + (ys[hi] - ys[lo]) * alpha;
}

PlotSeriesStyle plot_effective_series_style(const PreparedPlotSeries &series, PlotSeriesStyle pane_style) {
  if (pane_style != PlotSeriesStyle::Auto) return pane_style;
  return series.stairs ? PlotSeriesStyle::Step : PlotSeriesStyle::Line;
}

PlotYAxisBounds compute_plot_y_axis_bounds(const std::vector<PreparedPlotSeries> &series, const PlotYLimits &limits) {
  double auto_min = std::numeric_limits<double>::infinity();
  double auto_max = -std::numeric_limits<double>::infinity();
  for (const PreparedPlotSeries &item : series) {
    for (double value : item.ys) {
      if (!std::isfinite(value)) continue;
      auto_min = std::min(auto_min, value);
      auto_max = std::max(auto_max, value);
    }
  }

  if (!std::isfinite(auto_min) || !std::isfinite(auto_max)) {
    auto_min = 0.0;
    auto_max = 1.0;
  } else if (auto_max <= auto_min) {
    auto_min -= 0.5;
    auto_max += 0.5;
  } else {
    const double pad = (auto_max - auto_min) * 0.05;
    auto_min -= pad;
    auto_max += pad;
  }

  PlotYAxisBounds bounds;
  bounds.active = limits.active();
  bounds.min = limits.min_enabled ? limits.min : auto_min;
  bounds.max = limits.max_enabled ? limits.max : auto_max;
  if (!std::isfinite(bounds.min)) bounds.min = auto_min;
  if (!std::isfinite(bounds.max)) bounds.max = auto_max;
  if (bounds.max <= bounds.min) {
    if (limits.min_enabled && !limits.max_enabled) {
      bounds.max = bounds.min + 1.0;
    } else if (limits.max_enabled && !limits.min_enabled) {
      bounds.min = bounds.max - 1.0;
    } else {
      const double center = std::isfinite(bounds.min) ? bounds.min : 0.0;
      bounds.min = center - 0.5;
      bounds.max = center + 0.5;
    }
  }
  return bounds;
}

std::vector<PlotSeriesRequest> parse_plot_series_requests(std::string_view state_json) {
  std::string err;
  const json11::Json state = json11::Json::parse(std::string(state_json), err);
  std::vector<PlotSeriesRequest> out;

  auto add_request = [&](PlotSeriesRequest request) {
    if (request.path.empty()) return;
    if (request.label.empty()) request.label = plot_label_from_path(request.path);
    out.push_back(std::move(request));
  };

  auto add_path = [&](std::string path, std::string label = {}, bool stairs = false) {
    PlotSeriesRequest request;
    request.path = std::move(path);
    request.label = std::move(label);
    request.force_stairs = stairs;
    add_request(std::move(request));
  };

  auto add_json = [&](const json11::Json &item) {
    if (item.is_string()) {
      add_path(item.string_value());
    } else if (item.is_object()) {
      PlotSeriesRequest request;
      request.path = item["path"].is_string() ? item["path"].string_value() : item["name"].string_value();
      request.label = item["label"].is_string() ? item["label"].string_value() : item["title"].string_value();
      request.color = item["color"].is_string() ? item["color"].string_value() : std::string();
      request.transform = item["transform"].is_string() ? [item]() {
        if (item["transform"].string_value() == "derivative") return PlotSeriesTransform::Derivative;
        if (item["transform"].string_value() == "scale") return PlotSeriesTransform::Scale;
        return PlotSeriesTransform::None;
      }() : PlotSeriesTransform::None;
      request.derivative_dt = item["derivative_dt"].is_number() ? item["derivative_dt"].number_value() : 0.0;
      request.scale = item["scale"].is_number() ? item["scale"].number_value() : 1.0;
      request.offset = item["offset"].is_number() ? item["offset"].number_value() : 0.0;
      request.force_stairs = item["stairs"].bool_value() || item["step"].bool_value();
      request.custom_python = item["custom_python"].is_object() ? item["custom_python"] : json11::Json();
      add_request(std::move(request));
    }
  };

  if (!err.empty()) {
    add_path(std::string(state_json));
  } else if (state.is_string()) {
    add_path(state.string_value());
  } else if (state.is_object()) {
    if (state["path"].is_string()) add_path(state["path"].string_value(), state["label"].string_value(), state["stairs"].bool_value());
    const json11::Json &series = state["series"].is_array() ? state["series"] : state["paths"];
    for (const json11::Json &item : series.array_items()) add_json(item);
  } else if (state.is_array()) {
    for (const json11::Json &item : state.array_items()) add_json(item);
  }

  if (out.empty()) {
    add_path("/carState/vEgo", "vEgo");
    add_path("/carState/aEgo", "aEgo");
  }
  return out;
}

bool plot_has_series_path(const std::vector<PlotSeriesRequest> &requests, std::string_view path) {
  return std::any_of(requests.begin(), requests.end(), [&](const PlotSeriesRequest &request) {
    return request.path == path;
  });
}

json11::Json plot_series_request_json(const PlotSeriesRequest &request) {
  json11::Json::object item{
    {"path", request.path},
    {"label", request.label.empty() ? plot_label_from_path(request.path) : request.label},
  };
  if (!request.color.empty()) item["color"] = request.color;
  if (request.transform != PlotSeriesTransform::None) {
    const char *transform = request.transform == PlotSeriesTransform::Derivative ? "derivative" :
                           request.transform == PlotSeriesTransform::Scale ? "scale" : "";
    item["transform"] = transform;
    if (request.transform == PlotSeriesTransform::Derivative && request.derivative_dt > 0.0) {
      item["derivative_dt"] = request.derivative_dt;
    }
    if (request.transform == PlotSeriesTransform::Scale || request.scale != 1.0 || request.offset != 0.0) {
      item["scale"] = request.scale;
      item["offset"] = request.offset;
    }
  }
  if (request.custom_python.is_object()) item["custom_python"] = request.custom_python;
  if (request.force_stairs) item["stairs"] = true;
  return item;
}

void apply_plot_series_transform(const PlotSeriesRequest &request, std::vector<double> *xs, std::vector<double> *ys) {
  if (xs == nullptr || ys == nullptr || xs->size() != ys->size()) return;

  if (request.transform == PlotSeriesTransform::Derivative) {
    std::vector<double> transformed_xs;
    std::vector<double> transformed_ys;
    if (xs->size() >= 2) {
      transformed_xs.reserve(xs->size() - 1);
      transformed_ys.reserve(ys->size() - 1);
      for (size_t i = 1; i < xs->size(); ++i) {
        const double dt = request.derivative_dt > 0.0 ? request.derivative_dt : ((*xs)[i] - (*xs)[i - 1]);
        if (dt <= 0.0 || !std::isfinite(dt)) continue;
        transformed_xs.push_back((*xs)[i]);
        transformed_ys.push_back(((*ys)[i] - (*ys)[i - 1]) / dt);
      }
    }
    *xs = std::move(transformed_xs);
    *ys = std::move(transformed_ys);
  }

  if (request.transform == PlotSeriesTransform::Scale ||
      request.transform == PlotSeriesTransform::Derivative ||
      request.scale != 1.0 ||
      request.offset != 0.0) {
    for (double &value : *ys) value = value * request.scale + request.offset;
  }
}

// Common serializer for all pane-state mutations below: series list is always rebuilt from
// `requests`; the remaining fields (max_points, style, y_limits, zoom history) are each taken
// from whichever of `old_state`/override the specific caller wants (see wrappers).
std::string plot_state_json(const std::vector<PlotSeriesRequest> &requests, const json11::Json &old_state,
                            PlotSeriesStyle style, const PlotYLimits &limits,
                            const std::vector<PlotZoomRange> &zoom_history) {
  json11::Json::array series;
  series.reserve(requests.size());
  for (const PlotSeriesRequest &request : requests) {
    series.push_back(plot_series_request_json(request));
  }

  json11::Json::object out{{"series", series}};
  if (old_state.is_object() && old_state["max_points"].is_number()) {
    out["max_points"] = old_state["max_points"];
  }
  if (style != PlotSeriesStyle::Auto) out["style"] = plot_series_style_token(style);
  const json11::Json limits_json = plot_y_limits_json(limits);
  if (limits_json.is_object() && !limits_json.object_items().empty()) out["y_limits"] = limits_json;
  const json11::Json zoom_history_json = plot_zoom_history_json(zoom_history);
  if (zoom_history_json.is_array() && !zoom_history_json.array_items().empty()) out["x_zoom_history"] = zoom_history_json;
  return json11::Json(out).dump();
}

std::string plot_state_for_series(const std::vector<PlotSeriesRequest> &requests, const json11::Json &old_state) {
  return plot_state_json(requests, old_state, plot_series_style_node(old_state),
                        parse_plot_y_limits_node(old_state), parse_plot_zoom_history_node(old_state));
}

std::string plot_state_with_display_options(std::string_view state_json, PlotSeriesStyle style, const PlotYLimits &limits) {
  std::string err;
  const json11::Json old_state = json11::Json::parse(std::string(state_json), err);
  const std::vector<PlotSeriesRequest> requests = parse_plot_series_requests(state_json);
  return plot_state_json(requests, old_state, style, limits, parse_plot_zoom_history_node(old_state));
}

std::string plot_state_with_zoom_history(std::string_view state_json, const std::vector<PlotZoomRange> &history) {
  std::string err;
  const json11::Json old_state = json11::Json::parse(std::string(state_json), err);
  const std::vector<PlotSeriesRequest> requests = parse_plot_series_requests(state_json);
  return plot_state_json(requests, old_state, plot_series_style_node(old_state),
                        parse_plot_y_limits_node(old_state), history);
}

std::string plot_state_with_added_series(std::string_view state_json, std::string_view path) {
  if (path.empty()) return std::string(state_json);

  std::string err;
  const json11::Json old_state = json11::Json::parse(std::string(state_json), err);
  std::vector<PlotSeriesRequest> requests = parse_plot_series_requests(state_json);
  if (!plot_has_series_path(requests, path)) {
    requests.push_back({
      .path = std::string(path),
      .label = plot_label_from_path(path),
    });
  }
  return plot_state_for_series(requests, err.empty() ? old_state : json11::Json());
}

std::string plot_state_without_series(std::string_view state_json, std::string_view path) {
  if (path.empty()) return std::string(state_json);
  std::string err;
  const json11::Json old_state = json11::Json::parse(std::string(state_json), err);
  std::vector<PlotSeriesRequest> requests = parse_plot_series_requests(state_json);
  requests.erase(std::remove_if(requests.begin(), requests.end(),
                                [&](const PlotSeriesRequest &r) { return r.path == path; }), requests.end());
  return plot_state_for_series(requests, err.empty() ? old_state : json11::Json());
}

size_t parse_plot_max_points(std::string_view state_json, size_t fallback) {
  std::string err;
  const json11::Json state = json11::Json::parse(std::string(state_json), err);
  if (!err.empty() || !state.is_object() || !state["max_points"].is_number()) return fallback;
  return std::max<size_t>(64, static_cast<size_t>(state["max_points"].int_value()));
}

std::vector<PreparedPlotSeries> prepare_plot_series(const Store &store,
                                                   const std::vector<PlotSeriesRequest> &requests,
                                                   TimeRange range,
                                                   double tracker_time,
                                                   size_t max_points) {
  std::vector<PreparedPlotSeries> prepared;
  prepared.reserve(requests.size());
  for (const PlotSeriesRequest &request : requests) {
    PreparedPlotSeries item;
    item.request = request;
    item.view = store.series(request.path, range.start_, range.end, max_points);
    item.xs.reserve(item.view.points.size());
    item.ys.reserve(item.view.points.size());
    for (const SeriesPoint &point : item.view.points) {
      item.xs.push_back(point.t);
      item.ys.push_back(point.value);
    }
    apply_plot_series_transform(request, &item.xs, &item.ys);
    item.stairs = request.force_stairs || plot_integer_like(item.ys);
    if (!item.xs.empty() && item.xs.size() == item.ys.size()) {
      item.has_tracker_value = true;
      item.tracker_value = plot_sample_at_time(item.xs, item.ys, item.stairs, tracker_time);
    }
    prepared.push_back(std::move(item));
  }
  return prepared;
}

constexpr size_t kPlotZoomHistoryLimit = 16;
constexpr size_t kSeriesSelectorLimit = 500;

struct PlotTransientState {
  uint64_t series_generation = std::numeric_limits<uint64_t>::max();
  std::string series_filter;
  std::vector<std::string> series_paths;
};

PlotTransientState &plot_transient_state(PaneInstance &pane) {
  if (PlotTransientState *state = std::any_cast<PlotTransientState>(&pane.transient_state)) return *state;
  pane.transient_state = PlotTransientState{};
  return std::any_cast<PlotTransientState &>(pane.transient_state);
}

std::string legend_label(const PreparedPlotSeries &series, size_t label_width) {
  if (!series.has_tracker_value) return series.request.label;
  char value[64];
  std::snprintf(value, sizeof(value), "%.6g", series.tracker_value);
  const size_t width = std::max(label_width, series.request.label.size());
  return series.request.label + std::string(width - series.request.label.size() + 2, ' ') + value;
}

ImVec4 plot_color_for_request(const PlotSeriesRequest &request, size_t index) {
  unsigned int r = 0;
  unsigned int g = 0;
  unsigned int b = 0;
  if (request.color.size() == 7 && request.color[0] == '#' &&
      std::sscanf(request.color.c_str() + 1, "%02x%02x%02x", &r, &g, &b) == 3) {
    return ImVec4(static_cast<float>(r) / 255.0f,
                  static_cast<float>(g) / 255.0f,
                  static_cast<float>(b) / 255.0f,
                  1.0f);
  }
  const auto &palette = theme().plot_series_palette;
  return palette[index % palette.size()];
}

// Was a persistent toolbar row; now folded into the plot-hover tooltip (item 6 — the coverage
// line was chrome, not something worth a permanent row).
std::string plot_coverage_summary(const std::vector<PreparedPlotSeries> &series) {
  size_t visible = 0;
  size_t missing = 0;
  bool decimated = false;
  double covered = 0.0;
  for (const PreparedPlotSeries &item : series) {
    if (item.view.points.empty()) {
      ++missing;
    } else {
      ++visible;
      covered = std::max(covered, item.view.coverage.covered_seconds);
    }
    decimated = decimated || item.view.decimated;
  }

  char buffer[160];
  int written = std::snprintf(buffer, sizeof(buffer), "%zu series | %.2fs covered", visible, covered);
  std::string summary(buffer, static_cast<size_t>(std::max(0, written)));
  if (missing > 0) {
    std::snprintf(buffer, sizeof(buffer), " | %zu missing", missing);
    summary += buffer;
  }
  if (decimated) summary += " | decimated";
  return summary;
}

bool draw_plot_series_selector(const Store &store, PaneInstance *pane) {
  if (pane == nullptr) return false;
  bool changed = false;
  if (ImGui::GetContentRegionAvail().x > 100.0f) ImGui::SameLine();
  if (ImGui::Button("+ Series")) ImGui::OpenPopup("##plot_series_selector");

  if (ImGui::BeginPopup("##plot_series_selector")) {
    PlotTransientState &state = plot_transient_state(*pane);
    ImGui::SetNextItemWidth(340.0f);
    const bool filter_changed = input_text_with_hint("##plot_series_filter", "Search series", &state.series_filter);

    const std::vector<PlotSeriesRequest> requests = parse_plot_series_requests(pane->state_json);
    if (state.series_generation != store.generation() || filter_changed) {
      state.series_generation = store.generation();
      state.series_paths = store.series_paths_matching(state.series_filter, kSeriesSelectorLimit);
    }
    ImGui::TextDisabled("%zu loaded series", state.series_paths.size());
    ImGui::BeginChild("##plot_series_selector_rows", ImVec2(460.0f, 280.0f), true);
    size_t shown = 0;
    for (const std::string &path : state.series_paths) {
      ImGui::PushID(path.c_str());
      const bool present = plot_has_series_path(requests, path);
      if (present) ImGui::BeginDisabled();
      if (ImGui::SmallButton(present ? "Added" : "Add")) {
        pane->state_json = plot_state_with_added_series(pane->state_json, path);
        changed = true;
      }
      if (present) ImGui::EndDisabled();
      ImGui::SameLine();
      push_mono_font();
      ImGui::TextUnformatted(path.c_str());
      pop_mono_font();
      ImGui::PopID();
      ++shown;
    }
    if (shown == 0) ImGui::TextDisabled("No matching series");
    else if (shown >= kSeriesSelectorLimit) ImGui::TextDisabled("Showing first 500 matches");
    ImGui::EndChild();
    ImGui::EndPopup();
  }
  return changed;
}

// Chip: drag source (cross-plot drop) plus an "x" close button to remove the series.
bool draw_plot_series_drag_sources(PaneInstance *pane, const std::vector<PlotSeriesRequest> &requests) {
  bool removed = false;
  if (requests.empty() || pane == nullptr) return removed;
  for (size_t i = 0; i < requests.size(); ++i) {
    const PlotSeriesRequest &request = requests[i];
    if (i > 0 && ImGui::GetContentRegionAvail().x > 150.0f) ImGui::SameLine();
    ImGui::PushID(static_cast<int>(i));
    const std::string label = request.label.empty() ? plot_label_from_path(request.path) : request.label;
    const auto &palette = theme().plot_series_palette;
    ImGui::TextColored(palette[i % palette.size()], "%s", label.c_str());
    if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID)) {
      ImGui::SetDragDropPayload(kLoggySeriesPathPayload, request.path.c_str(), request.path.size() + 1);
      ImGui::TextUnformatted(request.path.c_str());
      ImGui::EndDragDropSource();
    }
    ImGui::SameLine(0.0f, 4.0f);
    if (ImGui::SmallButton("x")) {
      pane->state_json = plot_state_without_series(pane->state_json, request.path);
      removed = true;
    }
    ImGui::PopID();
  }
  return removed;
}

bool accept_series_drop(PaneInstance *pane) {
  bool changed = false;
  if (!ImGui::BeginDragDropTarget()) return false;
  if (const ImGuiPayload *payload = ImGui::AcceptDragDropPayload(kLoggySeriesPathPayload, ImGuiDragDropFlags_AcceptBeforeDelivery)) {
    if (payload->Preview) {
      const ImVec2 min = ImGui::GetItemRectMin();
      const ImVec2 max = ImGui::GetItemRectMax();
      ImDrawList *draw_list = ImGui::GetWindowDrawList();
      draw_list->AddRectFilled(min, max, ImGui::GetColorU32(theme().plot_drop_target_fill));
      draw_list->AddRect(min, max, ImGui::GetColorU32(theme().plot_drop_target_border), 0.0f, 0, 2.0f);
    }
    if (!payload->Delivery) {
      ImGui::EndDragDropTarget();
      return false;
    }
    if (payload->Data != nullptr && payload->DataSize > 0) {
      const char *data = static_cast<const char *>(payload->Data);
      const int size = data[payload->DataSize - 1] == '\0' ? payload->DataSize - 1 : payload->DataSize;
      if (size > 0) {
        pane->state_json = plot_state_with_added_series(pane->state_json, std::string_view(data, static_cast<size_t>(size)));
        changed = true;
      }
    }
  }
  ImGui::EndDragDropTarget();
  return changed;
}

int style_combo_index(PlotSeriesStyle style) {
  switch (style) {
    case PlotSeriesStyle::Line: return 1;
    case PlotSeriesStyle::Step: return 2;
    case PlotSeriesStyle::Scatter: return 3;
    case PlotSeriesStyle::Auto:
    default: return 0;
  }
}

PlotSeriesStyle style_from_combo_index(int index) {
  switch (index) {
    case 1: return PlotSeriesStyle::Line;
    case 2: return PlotSeriesStyle::Step;
    case 3: return PlotSeriesStyle::Scatter;
    case 0:
    default: return PlotSeriesStyle::Auto;
  }
}

bool draw_plot_display_controls(PlotSeriesStyle *style, PlotYLimits *limits) {
  if (style == nullptr || limits == nullptr) return false;

  bool changed = false;
  int style_index = style_combo_index(*style);
  ImGui::SetNextItemWidth(96.0f);
  if (ImGui::Combo("Style", &style_index, "Auto\0Line\0Step\0Scatter\0")) {
    *style = style_from_combo_index(style_index);
    changed = true;
  }

  ImGui::SameLine();
  if (ImGui::Button("Y")) ImGui::OpenPopup("##plot_y_limits");
  if (limits->active()) {
    ImGui::SameLine();
    if (limits->min_enabled && limits->max_enabled) {
      ImGui::TextDisabled("[%.6g, %.6g]", limits->min, limits->max);
    } else if (limits->min_enabled) {
      ImGui::TextDisabled("[%.6g, auto]", limits->min);
    } else {
      ImGui::TextDisabled("[auto, %.6g]", limits->max);
    }
  }

  if (ImGui::BeginPopup("##plot_y_limits")) {
    bool min_enabled = limits->min_enabled;
    if (ImGui::Checkbox("Min", &min_enabled)) {
      limits->min_enabled = min_enabled;
      changed = true;
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(120.0f);
    double min_value = limits->min;
    if (ImGui::InputDouble("##plot_y_min", &min_value, 0.0, 0.0, "%.9g")) {
      limits->min = min_value;
      limits->min_enabled = true;
      changed = true;
    }

    bool max_enabled = limits->max_enabled;
    if (ImGui::Checkbox("Max", &max_enabled)) {
      limits->max_enabled = max_enabled;
      changed = true;
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(120.0f);
    double max_value = limits->max;
    if (ImGui::InputDouble("##plot_y_max", &max_value, 0.0, 0.0, "%.9g")) {
      limits->max = max_value;
      limits->max_enabled = true;
      changed = true;
    }

    if (ImGui::Button("Apply")) {
      changed = true;
      ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset")) {
      *limits = {};
      changed = true;
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  }

  return changed;
}

bool ranges_near(PlotZoomRange lhs, PlotZoomRange rhs) {
  return std::abs(lhs.start_ - rhs.start_) <= 1.0e-6 && std::abs(lhs.end - rhs.end) <= 1.0e-6;
}

void push_zoom_history(std::vector<PlotZoomRange> *history, TimeRange range) {
  if (history == nullptr) return;
  PlotZoomRange snapshot{.start_ = range.start_, .end = range.end};
  if (!plot_zoom_range_valid(snapshot)) return;
  if (!history->empty() && ranges_near(history->back(), snapshot)) return;
  history->push_back(snapshot);
  while (history->size() > kPlotZoomHistoryLimit) history->erase(history->begin());
}

}  // namespace

void draw_plot_pane(Session &session, PaneInstance &pane) {
  // Compact toolbar: same controls (Style/Y/+Series/Undo Zoom), tighter padding so the chart
  // gets the vertical space the coverage line used to take (that line is now a hover tooltip).
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(5.0f, 2.0f));
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(6.0f, 2.0f));
  PlotSeriesStyle style = parse_plot_series_style(pane.state_json);
  PlotYLimits y_limits = parse_plot_y_limits(pane.state_json);
  if (draw_plot_display_controls(&style, &y_limits)) {
    pane.state_json = plot_state_with_display_options(pane.state_json, style, y_limits);
  }
  draw_plot_series_selector(session.store, &pane);

  std::vector<PlotZoomRange> zoom_history = parse_plot_zoom_history(pane.state_json);
  if (ImGui::GetContentRegionAvail().x > 112.0f) ImGui::SameLine();
  const bool undo_disabled = zoom_history.empty();
  if (undo_disabled) ImGui::BeginDisabled();
  if (ImGui::Button("Undo Zoom")) {
    const PlotZoomRange previous = zoom_history.back();
    zoom_history.pop_back();
    session.view_range.set_range({previous.start_, previous.end});
    pane.state_json = plot_state_with_zoom_history(pane.state_json, zoom_history);
  }
  if (undo_disabled) ImGui::EndDisabled();
  ImGui::PopStyleVar(2);

  const TimeRange range = session.view_range.range();

  const std::vector<PlotSeriesRequest> requests = parse_plot_series_requests(pane.state_json);
  const size_t pixel_cap = static_cast<size_t>(std::max(256.0f, ImGui::GetContentRegionAvail().x * 2.0f));
  const size_t max_points = parse_plot_max_points(pane.state_json, pixel_cap);
  std::vector<PreparedPlotSeries> series = prepare_plot_series(session.store, requests, range,
                                                               session.playback.tracker_time(), max_points);
  const PlotYAxisBounds y_bounds = compute_plot_y_axis_bounds(series, y_limits);

  const std::string coverage_summary = plot_coverage_summary(series);
  const bool series_removed = draw_plot_series_drag_sources(&pane, requests);

  bool has_points = false;
  size_t label_width = 0;
  for (const PreparedPlotSeries &item : series) {
    has_points = has_points || !item.xs.empty();
    label_width = std::max(label_width, item.request.label.size());
  }

  double x_min = range.start_;
  double x_max = range.end > range.start_ ? range.end : range.start_ + 1.0;
  ImPlotFlags plot_flags = ImPlotFlags_NoTitle | ImPlotFlags_NoMenus;
  if (!has_points) plot_flags |= ImPlotFlags_NoLegend;

  // All theme-constant ImPlot colors and legend padding/spacing are set once in apply_theme().
  const Theme &t = theme();
  push_mono_font();
  bool dropped_series = false;
  ImGui::BeginChild("##loggy_plot_child", ImGui::GetContentRegionAvail(), false,
                    ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
  if (ImPlot::BeginPlot("##loggy_plot", ImGui::GetContentRegionAvail(), plot_flags)) {
    ImPlot::SetupAxes(nullptr, nullptr,
                      ImPlotAxisFlags_NoMenus | ImPlotAxisFlags_NoHighlight,
                      ImPlotAxisFlags_NoMenus | ImPlotAxisFlags_NoHighlight | ImPlotAxisFlags_AutoFit);
    ImPlot::SetupAxisFormat(ImAxis_X1, "%.1f");
    ImPlot::SetupAxisFormat(ImAxis_Y1, "%.6g");
    ImPlot::SetupAxisLinks(ImAxis_X1, &x_min, &x_max);
    ImPlot::SetupAxisLimitsConstraints(ImAxis_X1, session.playback.route_range().start_,
                                       session.playback.route_range().end);
    if (y_bounds.active) ImPlot::SetupAxisLimits(ImAxis_Y1, y_bounds.min, y_bounds.max, ImPlotCond_Always);
    if (has_points) ImPlot::SetupLegend(ImPlotLocation_NorthEast);

    for (size_t i = 0; i < series.size(); ++i) {
      const PreparedPlotSeries &item = series[i];
      if (item.xs.size() < 2 || item.xs.size() != item.ys.size()) continue;

      const PlotSeriesStyle item_style = plot_effective_series_style(item, style);
      ImPlotSpec spec;
      spec.LineColor = plot_color_for_request(item.request, i);
      spec.LineWeight = item_style == PlotSeriesStyle::Step ? 1.8f : 2.0f;
      if (item_style == PlotSeriesStyle::Step) {
        spec.Flags = ImPlotStairsFlags_PreStep;
      } else if (item_style == PlotSeriesStyle::Scatter) {
        spec.Marker = ImPlotMarker_Circle;
        spec.MarkerSize = 3.0f;
        spec.MarkerFillColor = spec.LineColor;
        spec.LineWeight = 0.0f;
      } else {
        spec.Flags = ImPlotLineFlags_SkipNaN;
      }
      const std::string label = legend_label(item, label_width) + "###series_" + std::to_string(i) + "_" + item.request.path;
      if (item_style == PlotSeriesStyle::Step) {
        ImPlot::PlotStairs(label.c_str(), item.xs.data(), item.ys.data(), static_cast<int>(item.xs.size()), spec);
      } else if (item_style == PlotSeriesStyle::Scatter) {
        ImPlot::PlotScatter(label.c_str(), item.xs.data(), item.ys.data(), static_cast<int>(item.xs.size()), spec);
      } else {
        ImPlot::PlotLine(label.c_str(), item.xs.data(), item.ys.data(), static_cast<int>(item.xs.size()), spec);
      }
    }

    const double tracker = std::clamp(session.playback.tracker_time(),
                                      session.playback.route_range().start_,
                                      session.playback.route_range().end);
    ImPlotSpec tracker_spec;
    tracker_spec.LineColor = t.plot_tracker_line;
    tracker_spec.LineWeight = 1.0f;
    tracker_spec.Flags = ImPlotItemFlags_NoLegend;
    ImPlot::PlotInfLines("##tracker", &tracker, 1, tracker_spec);

    if (ImPlot::IsPlotHovered()) {
      const double hover_t = ImPlot::GetPlotMousePos().x;
      ImGui::BeginTooltip();
      ImGui::TextUnformatted(coverage_summary.c_str());
      ImGui::Separator();
      ImGui::Text("t %.3f", hover_t);
      for (const PreparedPlotSeries &item : series) {
        if (item.xs.empty() || item.xs.size() != item.ys.size()) continue;
        const bool stairs = plot_effective_series_style(item, style) == PlotSeriesStyle::Step;
        ImGui::Text("%s %.6g", item.request.label.c_str(), plot_sample_at_time(item.xs, item.ys, stairs, hover_t));
      }
      ImGui::EndTooltip();
    }

    if (ImPlot::IsPlotHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
      session.playback.seek(ImPlot::GetPlotMousePos().x);
    }
    ImPlot::EndPlot();
  }
  ImGui::EndChild();
  dropped_series = accept_series_drop(&pane);
  pop_mono_font();

  if (dropped_series || series_removed) return;

  if (std::abs(x_min - range.start_) > 1.0e-6 || std::abs(x_max - range.end) > 1.0e-6) {
    push_zoom_history(&zoom_history, range);
    pane.state_json = plot_state_with_zoom_history(pane.state_json, zoom_history);
    session.view_range.set_range({x_min, x_max});
  }
}

}  // namespace loggy
