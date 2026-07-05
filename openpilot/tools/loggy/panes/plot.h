#pragma once

#include "tools/loggy/backend/store.h"
#include "tools/loggy/shell/pane.h"
#include "tools/loggy/shell/transport.h"

#include "json11/json11.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace loggy {

struct PlotSeriesRequest {
  std::string path;
  std::string label;
  bool force_stairs = false;
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

inline std::string plot_label_from_path(std::string_view path) {
  if (path.empty()) return "series";
  const size_t slash = path.find_last_of('/');
  std::string label = slash == std::string_view::npos ? std::string(path) : std::string(path.substr(slash + 1));
  return label.empty() ? std::string(path) : label;
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

inline std::vector<PlotSeriesRequest> parse_plot_series_requests(std::string_view state_json) {
  std::string err;
  const json11::Json state = json11::Json::parse(std::string(state_json), err);
  std::vector<PlotSeriesRequest> out;

  auto add_path = [&](std::string path, std::string label = {}, bool stairs = false) {
    if (path.empty()) return;
    if (label.empty()) label = plot_label_from_path(path);
    out.push_back({
      .path = std::move(path),
      .label = std::move(label),
      .force_stairs = stairs,
    });
  };

  auto add_json = [&](const json11::Json &item) {
    if (item.is_string()) {
      add_path(item.string_value());
    } else if (item.is_object()) {
      const std::string path = item["path"].is_string() ? item["path"].string_value() : item["name"].string_value();
      const std::string label = item["label"].is_string() ? item["label"].string_value() : item["title"].string_value();
      add_path(path, label, item["stairs"].bool_value() || item["step"].bool_value());
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
  if (request.force_stairs) item["stairs"] = true;
  return item;
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
