#pragma once

#include "tools/loggy/backend/store.h"
#include "tools/loggy/shell/pane.h"

#include "json11/json11.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace loggy {

inline constexpr const char *kLoggySeriesPathPayload = "LOGGY_SERIES_PATH";

struct BrowserState {
  std::string filter;
  size_t max_rows = 1000;
  int sparkline_seconds = 30;
};

struct BrowserSparkline {
  std::vector<double> values;
  double min = 0.0;
  double max = 0.0;
};

struct BrowserSeriesRow {
  std::string path;
  std::string label;
  bool has_value = false;
  std::string value = "--";
  BrowserSparkline sparkline;
};

inline BrowserState parse_browser_state(std::string_view state_json) {
  BrowserState state;
  std::string err;
  const json11::Json json = json11::Json::parse(std::string(state_json), err);
  if (!err.empty() || !json.is_object()) return state;
  if (json["filter"].is_string()) state.filter = json["filter"].string_value();
  if (json["max_rows"].is_number()) {
    state.max_rows = static_cast<size_t>(std::clamp(json["max_rows"].int_value(), 1, 10000));
  }
  if (json["sparkline_seconds"].is_number()) {
    state.sparkline_seconds = std::clamp(json["sparkline_seconds"].int_value(), 1, 120);
  }
  return state;
}

inline std::string browser_state_json(const BrowserState &state) {
  return json11::Json(json11::Json::object{
    {"filter", state.filter},
    {"max_rows", static_cast<int>(state.max_rows)},
    {"sparkline_seconds", state.sparkline_seconds},
  }).dump();
}

inline std::string browser_leaf_label(std::string_view path) {
  if (path.empty()) return "series";
  const size_t slash = path.find_last_of('/');
  const std::string_view label = slash == std::string_view::npos ? path : path.substr(slash + 1);
  return label.empty() ? std::string(path) : std::string(label);
}

inline std::string browser_lower_text(std::string_view text) {
  std::string out(text);
  std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return out;
}

inline bool browser_path_matches_filter(std::string_view path, std::string_view filter) {
  if (filter.empty()) return true;
  const std::string haystack = browser_lower_text(path);
  const std::string needle = browser_lower_text(filter);
  return haystack.find(needle) != std::string::npos;
}

inline std::vector<BrowserSeriesRow> prepare_browser_series_rows(const Store &store, const BrowserState &state) {
  const std::vector<std::string> paths = store.seriesPaths();
  std::vector<BrowserSeriesRow> rows;
  rows.reserve(std::min(paths.size(), state.max_rows));
  for (const std::string &path : paths) {
    if (!browser_path_matches_filter(path, state.filter)) continue;
    rows.push_back({path, browser_leaf_label(path)});
    if (rows.size() >= state.max_rows) break;
  }
  return rows;
}

inline double browser_sample_at_time(const std::vector<SeriesPoint> &points, double time) {
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

inline std::string browser_format_value(double value) {
  if (!std::isfinite(value)) return "--";
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.6g", value);
  return buf;
}

inline BrowserSparkline browser_sparkline_from_view(const SeriesView &view,
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

inline BrowserSeriesRow enrich_browser_series_row(const Store &store,
                                                  BrowserSeriesRow row,
                                                  TimeRange range,
                                                  double tracker_time,
                                                  const BrowserState &state,
                                                  size_t max_points = 96) {
  const SeriesView view = store.series(row.path, range.start, range.end, max_points);
  if (!view.points.empty()) {
    row.has_value = true;
    row.value = browser_format_value(browser_sample_at_time(view.points, tracker_time));
    row.sparkline = browser_sparkline_from_view(view, 36, static_cast<double>(state.sparkline_seconds));
  }
  return row;
}

void draw_browser_pane(Session &session, PaneInstance &pane);

}  // namespace loggy
