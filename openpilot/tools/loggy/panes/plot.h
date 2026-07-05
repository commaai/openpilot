#pragma once

#include "tools/loggy/backend/store.h"
#include "tools/loggy/shell/pane.h"
#include "tools/loggy/shell/transport.h"

#include "json11/json11.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace loggy {

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

inline std::string plot_label_from_path(std::string_view path) {
  if (path.empty()) return "series";
  const size_t slash = path.find_last_of('/');
  std::string label = slash == std::string_view::npos ? std::string(path) : std::string(path.substr(slash + 1));
  return label.empty() ? std::string(path) : label;
}

inline PlotSeriesStyle plot_series_style_from_string(std::string_view style) {
  if (style == "line") return PlotSeriesStyle::Line;
  if (style == "step" || style == "stairs") return PlotSeriesStyle::Step;
  if (style == "scatter") return PlotSeriesStyle::Scatter;
  return PlotSeriesStyle::Auto;
}

inline const char *plot_series_style_token(PlotSeriesStyle style) {
  switch (style) {
    case PlotSeriesStyle::Line: return "line";
    case PlotSeriesStyle::Step: return "step";
    case PlotSeriesStyle::Scatter: return "scatter";
    case PlotSeriesStyle::Auto:
    default: return "auto";
  }
}

inline const char *plot_series_style_label(PlotSeriesStyle style) {
  switch (style) {
    case PlotSeriesStyle::Line: return "Line";
    case PlotSeriesStyle::Step: return "Step";
    case PlotSeriesStyle::Scatter: return "Scatter";
    case PlotSeriesStyle::Auto:
    default: return "Auto";
  }
}

inline PlotSeriesTransform plot_series_transform_from_string(std::string_view transform) {
  if (transform == "derivative") return PlotSeriesTransform::Derivative;
  if (transform == "scale") return PlotSeriesTransform::Scale;
  return PlotSeriesTransform::None;
}

inline const char *plot_series_transform_token(PlotSeriesTransform transform) {
  switch (transform) {
    case PlotSeriesTransform::Derivative: return "derivative";
    case PlotSeriesTransform::Scale: return "scale";
    case PlotSeriesTransform::None:
    default: return "";
  }
}

inline PlotSeriesStyle parse_plot_series_style_node(const json11::Json &state) {
  if (!state.is_object()) return PlotSeriesStyle::Auto;
  if (state["style"].is_string()) return plot_series_style_from_string(state["style"].string_value());
  if (state["series_type"].is_string()) return plot_series_style_from_string(state["series_type"].string_value());
  if (state["stairs"].bool_value() || state["step"].bool_value()) return PlotSeriesStyle::Step;
  return PlotSeriesStyle::Auto;
}

inline PlotSeriesStyle parse_plot_series_style(std::string_view state_json) {
  std::string err;
  const json11::Json state = json11::Json::parse(std::string(state_json), err);
  if (!err.empty()) return PlotSeriesStyle::Auto;
  return parse_plot_series_style_node(state);
}

inline PlotYLimits parse_plot_y_limits_node(const json11::Json &state) {
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

inline PlotYLimits parse_plot_y_limits(std::string_view state_json) {
  std::string err;
  const json11::Json state = json11::Json::parse(std::string(state_json), err);
  if (!err.empty()) return {};
  return parse_plot_y_limits_node(state);
}

inline json11::Json plot_y_limits_json(const PlotYLimits &limits) {
  json11::Json::object out;
  if (limits.min_enabled && std::isfinite(limits.min)) out["min"] = limits.min;
  if (limits.max_enabled && std::isfinite(limits.max)) out["max"] = limits.max;
  return out;
}

inline bool plot_integer_like(const std::vector<double> &ys) {
  if (ys.empty()) return false;
  for (double y : ys) {
    if (!std::isfinite(y)) continue;
    if (std::abs(y - std::round(y)) > 1.0e-9) return false;
  }
  return true;
}

inline double plot_sample_at_time(const std::vector<double> &xs, const std::vector<double> &ys, bool stairs, double time) {
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

inline PlotSeriesStyle plot_effective_series_style(const PreparedPlotSeries &series, PlotSeriesStyle pane_style) {
  if (pane_style != PlotSeriesStyle::Auto) return pane_style;
  return series.stairs ? PlotSeriesStyle::Step : PlotSeriesStyle::Line;
}

inline PlotYAxisBounds compute_plot_y_axis_bounds(const std::vector<PreparedPlotSeries> &series,
                                                  const PlotYLimits &limits) {
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

inline std::vector<PlotSeriesRequest> parse_plot_series_requests(std::string_view state_json) {
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
      request.transform = plot_series_transform_from_string(item["transform"].string_value());
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

inline bool plot_has_series_path(const std::vector<PlotSeriesRequest> &requests, std::string_view path) {
  return std::any_of(requests.begin(), requests.end(), [&](const PlotSeriesRequest &request) {
    return request.path == path;
  });
}

inline json11::Json plot_series_request_json(const PlotSeriesRequest &request) {
  json11::Json::object item{
    {"path", request.path},
    {"label", request.label.empty() ? plot_label_from_path(request.path) : request.label},
  };
  if (!request.color.empty()) item["color"] = request.color;
  if (request.transform != PlotSeriesTransform::None) {
    item["transform"] = plot_series_transform_token(request.transform);
    if (request.transform == PlotSeriesTransform::Derivative && request.derivative_dt > 0.0) {
      item["derivative_dt"] = request.derivative_dt;
    }
    if (request.transform == PlotSeriesTransform::Scale ||
        request.scale != 1.0 ||
        request.offset != 0.0) {
      item["scale"] = request.scale;
      item["offset"] = request.offset;
    }
  }
  if (request.custom_python.is_object()) item["custom_python"] = request.custom_python;
  if (request.force_stairs) item["stairs"] = true;
  return item;
}

inline void apply_plot_series_transform(const PlotSeriesRequest &request,
                                        std::vector<double> *xs,
                                        std::vector<double> *ys) {
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

inline std::string plot_state_for_series(const std::vector<PlotSeriesRequest> &requests, const json11::Json &old_state) {
  json11::Json::array series;
  series.reserve(requests.size());
  for (const PlotSeriesRequest &request : requests) {
    series.push_back(plot_series_request_json(request));
  }

  json11::Json::object out{{"series", series}};
  if (old_state.is_object() && old_state["max_points"].is_number()) {
    out["max_points"] = old_state["max_points"];
  }
  const PlotSeriesStyle style = parse_plot_series_style_node(old_state);
  if (style != PlotSeriesStyle::Auto) out["style"] = plot_series_style_token(style);
  const PlotYLimits limits = parse_plot_y_limits_node(old_state);
  const json11::Json limits_json = plot_y_limits_json(limits);
  if (limits_json.is_object() && !limits_json.object_items().empty()) out["y_limits"] = limits_json;
  return json11::Json(out).dump();
}

inline std::string plot_state_with_display_options(std::string_view state_json,
                                                   PlotSeriesStyle style,
                                                   const PlotYLimits &limits) {
  std::string err;
  const json11::Json old_state = json11::Json::parse(std::string(state_json), err);
  const std::vector<PlotSeriesRequest> requests = parse_plot_series_requests(state_json);

  json11::Json::array series;
  series.reserve(requests.size());
  for (const PlotSeriesRequest &request : requests) {
    series.push_back(plot_series_request_json(request));
  }

  json11::Json::object out{{"series", series}};
  if (err.empty() && old_state.is_object() && old_state["max_points"].is_number()) {
    out["max_points"] = old_state["max_points"];
  }
  if (style != PlotSeriesStyle::Auto) out["style"] = plot_series_style_token(style);
  const json11::Json limits_json = plot_y_limits_json(limits);
  if (limits_json.is_object() && !limits_json.object_items().empty()) out["y_limits"] = limits_json;
  return json11::Json(out).dump();
}

inline std::string plot_state_with_added_series(std::string_view state_json, std::string_view path) {
  if (path.empty()) return std::string(state_json);

  std::string err;
  const json11::Json old_state = json11::Json::parse(std::string(state_json), err);
  std::vector<PlotSeriesRequest> requests = parse_plot_series_requests(state_json);
  if (!plot_has_series_path(requests, path)) {
    requests.push_back({
      .path = std::string(path),
      .label = plot_label_from_path(path),
      .force_stairs = false,
    });
  }
  return plot_state_for_series(requests, err.empty() ? old_state : json11::Json());
}

inline size_t parse_plot_max_points(std::string_view state_json, size_t fallback) {
  std::string err;
  const json11::Json state = json11::Json::parse(std::string(state_json), err);
  if (!err.empty() || !state.is_object() || !state["max_points"].is_number()) return fallback;
  return std::max<size_t>(64, static_cast<size_t>(state["max_points"].int_value()));
}

inline std::vector<PreparedPlotSeries> prepare_plot_series(const Store &store,
                                                           const std::vector<PlotSeriesRequest> &requests,
                                                           TimeRange range,
                                                           double tracker_time,
                                                           size_t max_points) {
  std::vector<PreparedPlotSeries> prepared;
  prepared.reserve(requests.size());
  for (const PlotSeriesRequest &request : requests) {
    PreparedPlotSeries item;
    item.request = request;
    item.view = store.series(request.path, range.start, range.end, max_points);
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

void draw_plot_pane(Session &session, PaneInstance &pane);

}  // namespace loggy
