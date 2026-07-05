#pragma once

#include "tools/loggy/backend/store.h"
#include "tools/loggy/shell/pane.h"

#include "json11/json11.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace loggy {

inline constexpr const char *kDefaultMapLatitudePath = "/gpsLocationExternal/latitude";
inline constexpr const char *kDefaultMapLongitudePath = "/gpsLocationExternal/longitude";
inline constexpr const char *kDefaultMapHasFixPath = "/gpsLocationExternal/hasFix";
inline constexpr const char *kDefaultMapBearingPath = "/gpsLocationExternal/bearingDeg";

struct MapState {
  std::string latitude_path = kDefaultMapLatitudePath;
  std::string longitude_path = kDefaultMapLongitudePath;
  std::string has_fix_path = kDefaultMapHasFixPath;
  std::string bearing_path = kDefaultMapBearingPath;
  size_t max_points = 4000;
};

struct MapTracePoint {
  double t = 0.0;
  double lat = 0.0;
  double lon = 0.0;
  double bearing_deg = 0.0;
};

struct MapTrace {
  std::vector<MapTracePoint> points;
  double min_lat = 0.0;
  double max_lat = 0.0;
  double min_lon = 0.0;
  double max_lon = 0.0;
  bool decimated = false;

  bool valid() const { return !points.empty() && max_lat >= min_lat && max_lon >= min_lon; }
};

inline MapState parse_map_state(std::string_view state_json) {
  MapState state;
  std::string err;
  const json11::Json json = json11::Json::parse(std::string(state_json), err);
  if (!err.empty() || !json.is_object()) return state;
  if (json["latitude_path"].is_string()) state.latitude_path = json["latitude_path"].string_value();
  if (json["longitude_path"].is_string()) state.longitude_path = json["longitude_path"].string_value();
  if (json["has_fix_path"].is_string()) state.has_fix_path = json["has_fix_path"].string_value();
  if (json["bearing_path"].is_string()) state.bearing_path = json["bearing_path"].string_value();
  if (json["max_points"].is_number()) state.max_points = static_cast<size_t>(std::clamp(json["max_points"].int_value(), 64, 20000));
  return state;
}

inline bool valid_map_coordinate(double lat, double lon) {
  return std::isfinite(lat) && std::isfinite(lon) &&
         lat >= -90.0 && lat <= 90.0 &&
         lon >= -180.0 && lon <= 180.0 &&
         (std::abs(lat) > 1.0e-9 || std::abs(lon) > 1.0e-9);
}

inline double map_value_at_index_or_default(const SeriesView &view, size_t index, double fallback) {
  return index < view.points.size() ? view.points[index].value : fallback;
}

inline MapTrace prepare_map_trace(const Store &store, TimeRange range, const MapState &state) {
  MapTrace trace;
  const SeriesView lat = store.series(state.latitude_path, range.start, range.end, state.max_points);
  const SeriesView lon = store.series(state.longitude_path, range.start, range.end, state.max_points);
  const SeriesView fix = state.has_fix_path.empty() ? SeriesView{} : store.series(state.has_fix_path, range.start, range.end, state.max_points);
  const SeriesView bearing = state.bearing_path.empty() ? SeriesView{} : store.series(state.bearing_path, range.start, range.end, state.max_points);
  trace.decimated = lat.decimated || lon.decimated || fix.decimated || bearing.decimated;

  const size_t count = std::min(lat.points.size(), lon.points.size());
  trace.points.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    const double lat_value = lat.points[i].value;
    const double lon_value = lon.points[i].value;
    if (!valid_map_coordinate(lat_value, lon_value)) continue;
    if (!fix.points.empty() && map_value_at_index_or_default(fix, i, 1.0) <= 0.5) continue;

    MapTracePoint point;
    point.t = lat.points[i].t;
    point.lat = lat_value;
    point.lon = lon_value;
    point.bearing_deg = map_value_at_index_or_default(bearing, i, 0.0);
    trace.points.push_back(point);
  }
  if (trace.points.empty()) return trace;

  trace.min_lat = trace.max_lat = trace.points.front().lat;
  trace.min_lon = trace.max_lon = trace.points.front().lon;
  for (const MapTracePoint &point : trace.points) {
    trace.min_lat = std::min(trace.min_lat, point.lat);
    trace.max_lat = std::max(trace.max_lat, point.lat);
    trace.min_lon = std::min(trace.min_lon, point.lon);
    trace.max_lon = std::max(trace.max_lon, point.lon);
  }
  return trace;
}

inline std::optional<MapTracePoint> map_trace_point_at_time(const MapTrace &trace, double time) {
  if (trace.points.empty()) return std::nullopt;
  const auto upper = std::lower_bound(trace.points.begin(), trace.points.end(), time, [](const MapTracePoint &point, double t) {
    return point.t < t;
  });
  if (upper == trace.points.begin()) return trace.points.front();
  if (upper == trace.points.end()) return trace.points.back();
  const MapTracePoint &next = *upper;
  const MapTracePoint &prev = *(upper - 1);
  return std::abs(next.t - time) < std::abs(time - prev.t) ? next : prev;
}

void draw_map_pane(Session &session, PaneInstance &pane);

}  // namespace loggy
