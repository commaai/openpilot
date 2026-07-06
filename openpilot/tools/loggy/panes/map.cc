#include "tools/loggy/panes/map.h"

#include "tools/loggy/backend/session.h"
#include "tools/loggy/shell/native_dialog.h"
#include "tools/loggy/shell/theme.h"
#include "tools/loggy/shell/workspace.h"

#include "common/util.h"
#include "imgui.h"
#include "json11/json11.hpp"

#include <algorithm>
#include <any>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace loggy {
inline constexpr const char *kDefaultMapLatitudePath = "/gpsLocationExternal/latitude";
inline constexpr const char *kDefaultMapLongitudePath = "/gpsLocationExternal/longitude";
inline constexpr const char *kDefaultMapHasFixPath = "/gpsLocationExternal/hasFix";
inline constexpr const char *kDefaultMapBearingPath = "/gpsLocationExternal/bearingDeg";
inline constexpr double kMinMapZoom = 0.25;
inline constexpr double kMaxMapZoom = 32.0;

struct MapState {
  std::string latitude_path = kDefaultMapLatitudePath;
  std::string longitude_path = kDefaultMapLongitudePath;
  std::string has_fix_path = kDefaultMapHasFixPath;
  std::string bearing_path = kDefaultMapBearingPath;
  size_t max_points = 4000;
  bool follow = false;
  bool show_basemap = true;
  double zoom = 1.0;
  bool has_center = false;
  double center_lat = 0.0;
  double center_lon = 0.0;
};

struct MapTracePoint {
  double t = 0.0;
  double lat = 0.0;
  double lon = 0.0;
  double bearing_deg = 0.0;
  TimelineSpanKind kind = TimelineSpanKind::None;
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

struct MapBasemapBounds {
  double south = 0.0;
  double west = 0.0;
  double north = 0.0;
  double east = 0.0;

  bool valid() const { return south < north && west < east; }
};

struct MapBasemapPoint {
  double lat = 0.0;
  double lon = 0.0;
};

enum class MapBasemapFeatureKind {
  RoadMotorway,
  RoadPrimary,
  RoadSecondary,
  RoadLocal,
  WaterLine,
  WaterPolygon,
};

struct MapBasemapFeature {
  MapBasemapFeatureKind kind = MapBasemapFeatureKind::RoadLocal;
  MapBasemapBounds bounds;
  std::vector<MapBasemapPoint> points;
};

struct MapBasemap {
  std::string key;
  MapBasemapBounds bounds;
  std::vector<MapBasemapFeature> features;
};

struct MapBasemapRequest {
  std::string key;
  MapBasemapBounds bounds;
  std::string query;
};

struct MapBasemapCacheStats {
  uint64_t bytes = 0;
  size_t files = 0;
};

struct MapBasemapStatus {
  bool loading = false;
  bool fetching = false;
  std::string key;
  std::string message;
  MapBasemapCacheStats cache;
};

struct MapBasemapManager;

namespace {

namespace fs = std::filesystem;

constexpr double kMapTracePadFrac = 0.45;
constexpr double kMapTraceMinLatPad = 0.01;
constexpr double kMapBoundsGrid = 0.005;
constexpr double kMapCorridorLatPad = 0.010;
constexpr double kMapCorridorMinStepS = 1.5;
constexpr size_t kMapCorridorMaxBoxes = 36;
constexpr const char *kMapQueryEndpoints[] = {
  "https://overpass-api.de/api/interpreter",
  "https://overpass.private.coffee/api/interpreter",
};

struct MapProjection {
  double min_x = 0.0;
  double min_y = 0.0;
  double scale = 1.0;
  ImVec2 origin;
  ImVec2 size;
  double cos_lat = 1.0;
};

struct MapPaneTransientState {
  MapState state;
  std::string loaded_json;
  std::shared_ptr<MapBasemapManager> basemap;
};

MapState parse_map_state(std::string_view state_json);
std::string map_state_json(const MapState &state);

MapPaneTransientState &map_pane_transient_state(PaneInstance &pane) {
  if (auto *transient = std::any_cast<MapPaneTransientState>(&pane.transient_state)) return *transient;
  pane.transient_state = MapPaneTransientState{};
  return std::any_cast<MapPaneTransientState &>(pane.transient_state);
}

MapState &map_pane_state(PaneInstance &pane) {
  MapPaneTransientState &transient = map_pane_transient_state(pane);
  if (transient.loaded_json != pane.state_json) {
    transient.state = parse_map_state(pane.state_json);
    transient.loaded_json = pane.state_json;
  }
  return transient.state;
}

void save_map_pane_state(PaneInstance &pane, const MapState &state) {
  pane.state_json = map_state_json(state);
  MapPaneTransientState &transient = map_pane_transient_state(pane);
  transient.state = state;
  transient.loaded_json = pane.state_json;
}

MapState parse_map_state(std::string_view state_json) {
  MapState state;
  std::string err;
  const json11::Json json = json11::Json::parse(std::string(state_json), err);
  if (!err.empty() || !json.is_object()) return state;
  if (json["latitude_path"].is_string()) state.latitude_path = json["latitude_path"].string_value();
  if (json["longitude_path"].is_string()) state.longitude_path = json["longitude_path"].string_value();
  if (json["has_fix_path"].is_string()) state.has_fix_path = json["has_fix_path"].string_value();
  if (json["bearing_path"].is_string()) state.bearing_path = json["bearing_path"].string_value();
  if (json["max_points"].is_number()) state.max_points = static_cast<size_t>(std::clamp(json["max_points"].int_value(), 64, 20000));
  if (json["follow"].is_bool()) state.follow = json["follow"].bool_value();
  if (json["show_basemap"].is_bool()) state.show_basemap = json["show_basemap"].bool_value();
  if (json["zoom"].is_number() && std::isfinite(json["zoom"].number_value())) {
    state.zoom = std::clamp(json["zoom"].number_value(), kMinMapZoom, kMaxMapZoom);
  }
  if (json["center_lat"].is_number() && json["center_lon"].is_number()) {
    const double lat = json["center_lat"].number_value();
    const double lon = json["center_lon"].number_value();
    if (std::isfinite(lat) && std::isfinite(lon)) {
      state.has_center = true;
      state.center_lat = std::clamp(lat, -90.0, 90.0);
      state.center_lon = std::clamp(lon, -180.0, 180.0);
    }
  }
  return state;
}

std::string map_state_json(const MapState &state) {
  json11::Json::object out{
    {"latitude_path", state.latitude_path},
    {"longitude_path", state.longitude_path},
    {"has_fix_path", state.has_fix_path},
    {"bearing_path", state.bearing_path},
    {"max_points", static_cast<int>(state.max_points)},
    {"follow", state.follow},
    {"show_basemap", state.show_basemap},
    {"zoom", std::clamp(state.zoom, kMinMapZoom, kMaxMapZoom)},
  };
  if (state.has_center) {
    out["center_lat"] = std::clamp(state.center_lat, -90.0, 90.0);
    out["center_lon"] = std::clamp(state.center_lon, -180.0, 180.0);
  }
  return json11::Json(out).dump();
}

bool valid_map_coordinate(double lat, double lon) {
  return std::isfinite(lat) && std::isfinite(lon) &&
         lat >= -90.0 && lat <= 90.0 &&
         lon >= -180.0 && lon <= 180.0 &&
         (std::abs(lat) > 1.0e-9 || std::abs(lon) > 1.0e-9);
}

double map_value_at_index_or_default(const SeriesView &view, size_t index, double fallback) {
  return index < view.points.size() ? view.points[index].value : fallback;
}

MapTrace prepare_map_trace(const Store &store, TimeRange range, const MapState &state,
                          const TimelineModel *timeline) {
  MapTrace trace;
  const SeriesView lat = store.series(state.latitude_path, range.start_, range.end, state.max_points);
  const SeriesView lon = store.series(state.longitude_path, range.start_, range.end, state.max_points);
  const SeriesView fix = state.has_fix_path.empty() ? SeriesView{} : store.series(state.has_fix_path, range.start_, range.end, state.max_points);
  const SeriesView bearing = state.bearing_path.empty() ? SeriesView{} : store.series(state.bearing_path, range.start_, range.end, state.max_points);
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
    point.kind = timeline != nullptr ? timeline->kind_at_time(point.t) : TimelineSpanKind::None;
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

std::optional<MapTracePoint> map_trace_point_at_time(const MapTrace &trace, double time) {
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

std::string map_google_maps_url(const MapTrace &trace) {
  if (!trace.valid()) return {};
  const MapTracePoint &start_ = trace.points.front();
  const MapTracePoint &end = trace.points.back();
  char buf[256];
  std::snprintf(buf, sizeof(buf),
                "https://www.google.com/maps/dir/%.6f,%.6f/%.6f,%.6f",
                start_.lat, start_.lon, end.lat, end.lon);
  return buf;
}

double clamp_basemap_lat(double lat) {
  return std::clamp(lat, -85.0, 85.0);
}

double clamp_basemap_lon(double lon) {
  return std::clamp(lon, -179.999, 179.999);
}

double quantize_down(double value, double step) {
  return std::floor(value / step) * step;
}

double quantize_up(double value, double step) {
  return std::ceil(value / step) * step;
}

double map_cos_lat_scale(double lat) {
  return std::max(0.20, std::cos(lat * M_PI / 180.0));
}

MapBasemapBounds merge_basemap_bounds(const MapBasemapBounds &a, const MapBasemapBounds &b) {
  if (!a.valid()) return b;
  if (!b.valid()) return a;
  return {
    .south = std::min(a.south, b.south),
    .west = std::min(a.west, b.west),
    .north = std::max(a.north, b.north),
    .east = std::max(a.east, b.east),
  };
}

bool basemap_bounds_overlap_or_touch(const MapBasemapBounds &a, const MapBasemapBounds &b) {
  return !(a.east < b.west || b.east < a.west || a.north < b.south || b.north < a.south);
}

MapBasemapBounds feature_bounds(const std::vector<MapBasemapPoint> &points) {
  MapBasemapBounds bounds;
  if (points.empty()) return bounds;
  bounds.south = bounds.north = points.front().lat;
  bounds.west = bounds.east = points.front().lon;
  for (const MapBasemapPoint &point : points) {
    bounds.south = std::min(bounds.south, point.lat);
    bounds.north = std::max(bounds.north, point.lat);
    bounds.west = std::min(bounds.west, point.lon);
    bounds.east = std::max(bounds.east, point.lon);
  }
  return bounds;
}

std::vector<MapBasemapBounds> corridor_boxes_for_trace(const MapTrace &trace) {
  std::vector<MapBasemapBounds> boxes;
  if (!trace.valid()) return boxes;

  const double center_lat = (trace.min_lat + trace.max_lat) * 0.5;
  const double lon_pad = kMapCorridorLatPad / map_cos_lat_scale(center_lat);
  const double total_time = trace.points.back().t - trace.points.front().t;
  const double target_boxes = std::min<double>(kMapCorridorMaxBoxes, std::max<double>(8.0, total_time / kMapCorridorMinStepS));
  const size_t stride = std::max<size_t>(1, static_cast<size_t>(std::ceil(static_cast<double>(trace.points.size()) / target_boxes)));

  auto add_box = [&](double lat, double lon) {
    MapBasemapBounds box{
      .south = clamp_basemap_lat(quantize_down(lat - kMapCorridorLatPad, kMapBoundsGrid)),
      .west = clamp_basemap_lon(quantize_down(lon - lon_pad, kMapBoundsGrid)),
      .north = clamp_basemap_lat(quantize_up(lat + kMapCorridorLatPad, kMapBoundsGrid)),
      .east = clamp_basemap_lon(quantize_up(lon + lon_pad, kMapBoundsGrid)),
    };
    if (!box.valid()) return;
    for (MapBasemapBounds &existing : boxes) {
      if (basemap_bounds_overlap_or_touch(existing, box)) {
        existing = merge_basemap_bounds(existing, box);
        return;
      }
    }
    boxes.push_back(box);
  };

  add_box(trace.points.front().lat, trace.points.front().lon);
  for (size_t i = stride; i < trace.points.size(); i += stride) {
    add_box(trace.points[i].lat, trace.points[i].lon);
  }
  add_box(trace.points.back().lat, trace.points.back().lon);

  bool merged = true;
  while (merged) {
    merged = false;
    for (size_t i = 0; i < boxes.size() && !merged; ++i) {
      for (size_t j = i + 1; j < boxes.size(); ++j) {
        if (basemap_bounds_overlap_or_touch(boxes[i], boxes[j])) {
          boxes[i] = merge_basemap_bounds(boxes[i], boxes[j]);
          boxes.erase(boxes.begin() + static_cast<std::ptrdiff_t>(j));
          merged = true;
          break;
        }
      }
    }
  }
  return boxes;
}

std::string bbox_string(const MapBasemapBounds &bounds) {
  char buf[128];
  std::snprintf(buf, sizeof(buf), "%.6f,%.6f,%.6f,%.6f", bounds.south, bounds.west, bounds.north, bounds.east);
  return buf;
}

std::string bounds_key(const MapBasemapBounds &bounds) {
  char buf[160];
  std::snprintf(buf, sizeof(buf), "v1_%.5f_%.5f_%.5f_%.5f", bounds.south, bounds.west, bounds.north, bounds.east);
  return buf;
}

std::string feature_kind_token(MapBasemapFeatureKind kind) {
  switch (kind) {
    case MapBasemapFeatureKind::RoadMotorway: return "road_motorway";
    case MapBasemapFeatureKind::RoadPrimary: return "road_primary";
    case MapBasemapFeatureKind::RoadSecondary: return "road_secondary";
    case MapBasemapFeatureKind::RoadLocal: return "road_local";
    case MapBasemapFeatureKind::WaterLine: return "water_line";
    case MapBasemapFeatureKind::WaterPolygon: return "water_polygon";
  }
  return "road_local";
}

std::optional<MapBasemapFeatureKind> parse_feature_kind(std::string_view token) {
  if (token == "road_motorway") return MapBasemapFeatureKind::RoadMotorway;
  if (token == "road_primary") return MapBasemapFeatureKind::RoadPrimary;
  if (token == "road_secondary") return MapBasemapFeatureKind::RoadSecondary;
  if (token == "road_local") return MapBasemapFeatureKind::RoadLocal;
  if (token == "water_line") return MapBasemapFeatureKind::WaterLine;
  if (token == "water_polygon") return MapBasemapFeatureKind::WaterPolygon;
  return std::nullopt;
}

std::optional<MapBasemapFeatureKind> classify_road(std::string_view highway) {
  if (highway == "motorway" || highway == "motorway_link" || highway == "trunk" || highway == "trunk_link") {
    return MapBasemapFeatureKind::RoadMotorway;
  }
  if (highway == "primary" || highway == "primary_link") {
    return MapBasemapFeatureKind::RoadPrimary;
  }
  if (highway == "secondary" || highway == "secondary_link" || highway == "tertiary" || highway == "tertiary_link") {
    return MapBasemapFeatureKind::RoadSecondary;
  }
  if (highway == "residential" || highway == "unclassified" || highway == "living_street" || highway == "road") {
    return MapBasemapFeatureKind::RoadLocal;
  }
  return std::nullopt;
}

std::vector<MapBasemapPoint> geometry_points(const json11::Json &geometry_json) {
  std::vector<MapBasemapPoint> points;
  const auto items = geometry_json.array_items();
  points.reserve(items.size());
  for (const json11::Json &point : items) {
    if (!point["lat"].is_number() || !point["lon"].is_number()) continue;
    const double lat = point["lat"].number_value();
    const double lon = point["lon"].number_value();
    if (valid_map_coordinate(lat, lon)) points.push_back({.lat = lat, .lon = lon});
  }
  return points;
}

json11::Json bounds_json(const MapBasemapBounds &bounds) {
  return json11::Json::object{
    {"south", bounds.south},
    {"west", bounds.west},
    {"north", bounds.north},
    {"east", bounds.east},
  };
}

std::optional<MapBasemapBounds> parse_bounds_json(const json11::Json &json) {
  if (!json.is_object() || !json["south"].is_number() || !json["west"].is_number() ||
      !json["north"].is_number() || !json["east"].is_number()) {
    return std::nullopt;
  }
  MapBasemapBounds bounds{
    .south = json["south"].number_value(),
    .west = json["west"].number_value(),
    .north = json["north"].number_value(),
    .east = json["east"].number_value(),
  };
  return bounds.valid() ? std::optional<MapBasemapBounds>(bounds) : std::nullopt;
}

uint64_t fnv1a64(std::string_view text) {
  uint64_t value = 1469598103934665603ULL;
  for (unsigned char c : text) {
    value ^= static_cast<uint64_t>(c);
    value *= 1099511628211ULL;
  }
  return value;
}

std::string percent_encode(std::string_view text) {
  std::string out;
  out.reserve(text.size() * 3);
  for (unsigned char c : text) {
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
        c == '-' || c == '_' || c == '.' || c == '~') {
      out.push_back(static_cast<char>(c));
    } else {
      char buf[4];
      std::snprintf(buf, sizeof(buf), "%%%02X", static_cast<unsigned int>(c));
      out += buf;
    }
  }
  return out;
}

std::optional<std::string> fetch_overpass_json(std::string_view query) {
  const std::string body = std::string("data=") + percent_encode(query);
  for (const char *endpoint : kMapQueryEndpoints) {
    const std::string command = "curl -fsSL --compressed --connect-timeout 6 --max-time 20 "
                                "-A 'loggy-vector-map/1.0' "
                                "-H 'Content-Type: application/x-www-form-urlencoded; charset=UTF-8' "
                                "--data-raw " + shell_quote(body) + " " + shell_quote(endpoint);
    const std::string response = util::check_output(command);
    if (!response.empty() && response.front() == '{') {
      return response;
    }
  }
  return std::nullopt;
}

bool write_text_file(const fs::path &path, std::string_view text, std::string &error) {
  error.clear();
  std::error_code ec;
  if (!path.parent_path().empty()) {
    fs::create_directories(path.parent_path(), ec);
    if (ec) {
      error = ec.message();
      return false;
    }
  }
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out.is_open()) {
    error = "failed to open " + path.string();
    return false;
  }
  out << text;
  if (!out.good()) {
    error = "failed to write " + path.string();
    return false;
  }
  return true;
}

std::optional<std::string> read_text_file(const fs::path &path, std::string &error) {
  error.clear();
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    error = "failed to open " + path.string();
    return std::nullopt;
  }
  std::ostringstream out;
  out << in.rdbuf();
  if (!in.good() && !in.eof()) {
    error = "failed to read " + path.string();
    return std::nullopt;
  }
  return out.str();
}

double projected_x(double lon, double cos_lat) {
  return lon * cos_lat;
}

double projected_y(double lat) {
  return -lat;
}

MapProjection make_projection(const MapTrace &trace, ImVec2 origin, ImVec2 size,
                              const MapState &state,
                              const std::optional<MapTracePoint> &tracker) {
  double center_lat = (trace.min_lat + trace.max_lat) * 0.5;
  double center_lon = (trace.min_lon + trace.max_lon) * 0.5;
  if (state.follow && tracker.has_value()) {
    center_lat = tracker->lat;
    center_lon = tracker->lon;
  } else if (state.has_center) {
    center_lat = state.center_lat;
    center_lon = state.center_lon;
  }
  const double cos_lat = std::max(0.20, std::cos(center_lat * M_PI / 180.0));
  const double min_x = projected_x(trace.min_lon, cos_lat);
  const double max_x = projected_x(trace.max_lon, cos_lat);
  const double min_y = projected_y(trace.max_lat);
  const double max_y = projected_y(trace.min_lat);
  const double span_x = std::max(max_x - min_x, 1.0e-6);
  const double span_y = std::max(max_y - min_y, 1.0e-6);
  const double fit_scale = 0.88 * std::min(static_cast<double>(size.x) / span_x, static_cast<double>(size.y) / span_y);
  const double scale = fit_scale * std::clamp(state.zoom, kMinMapZoom, kMaxMapZoom);
  const double center_x = projected_x(center_lon, cos_lat);
  const double center_y = projected_y(center_lat);
  return {
    .min_x = center_x - static_cast<double>(size.x) * 0.5 / scale,
    .min_y = center_y - static_cast<double>(size.y) * 0.5 / scale,
    .scale = scale,
    .origin = origin,
    .size = size,
    .cos_lat = cos_lat,
  };
}

ImVec2 map_to_screen(const MapProjection &projection, double lat, double lon) {
  return ImVec2(
    projection.origin.x + static_cast<float>((projected_x(lon, projection.cos_lat) - projection.min_x) * projection.scale),
    projection.origin.y + static_cast<float>((projected_y(lat) - projection.min_y) * projection.scale));
}

void set_map_center_from_projected(MapState *state, double center_x, double center_y, double cos_lat) {
  if (state == nullptr) return;
  state->has_center = true;
  state->center_lat = std::clamp(-center_y, -90.0, 90.0);
  state->center_lon = std::clamp(center_x / std::max(0.20, cos_lat), -180.0, 180.0);
}

void set_map_center_from_projection(MapState *state, const MapProjection &projection) {
  set_map_center_from_projected(state,
                                projection.min_x + static_cast<double>(projection.size.x) * 0.5 / projection.scale,
                                projection.min_y + static_cast<double>(projection.size.y) * 0.5 / projection.scale,
                                projection.cos_lat);
}

void draw_map_grid(ImDrawList *draw_list, ImVec2 origin, ImVec2 size) {
  const ImU32 bg = ImGui::GetColorU32(color_rgb(43, 45, 47));
  const ImU32 border = ImGui::GetColorU32(color_rgb(92, 96, 98));
  const ImU32 major = ImGui::GetColorU32(color_rgb(72, 77, 80, 0.76f));
  draw_list->AddRectFilled(origin, ImVec2(origin.x + size.x, origin.y + size.y), bg);
  for (int i = 1; i < 6; ++i) {
    const float x = origin.x + size.x * static_cast<float>(i) / 6.0f;
    const float y = origin.y + size.y * static_cast<float>(i) / 6.0f;
    draw_list->AddLine(ImVec2(x, origin.y), ImVec2(x, origin.y + size.y), major);
    draw_list->AddLine(ImVec2(origin.x, y), ImVec2(origin.x + size.x, y), major);
  }
  draw_list->AddRect(origin, ImVec2(origin.x + size.x, origin.y + size.y), border);
}

void draw_car_marker(ImDrawList *draw_list, ImVec2 center, double bearing_deg, ImU32 color) {
  const float rad = static_cast<float>((bearing_deg - 90.0) * M_PI / 180.0);
  const ImVec2 forward(std::cos(rad), std::sin(rad));
  const ImVec2 right(-forward.y, forward.x);
  const float size = 8.0f;
  const ImVec2 p0(center.x + forward.x * size * 1.5f, center.y + forward.y * size * 1.5f);
  const ImVec2 p1(center.x - forward.x * size + right.x * size, center.y - forward.y * size + right.y * size);
  const ImVec2 p2(center.x - forward.x * size - right.x * size, center.y - forward.y * size - right.y * size);
  draw_list->AddTriangleFilled(p0, p1, p2, color);
  draw_list->AddTriangle(p0, p1, p2, ImGui::GetColorU32(color_rgb(30, 32, 34)), 1.5f);
}

int map_kind_priority(TimelineSpanKind kind) {
  switch (kind) {
    case TimelineSpanKind::AlertCritical:
      return 4;
    case TimelineSpanKind::AlertWarning:
      return 3;
    case TimelineSpanKind::AlertInfo:
      return 2;
    case TimelineSpanKind::Engaged:
      return 1;
    case TimelineSpanKind::None:
    default:
      return 0;
  }
}

TimelineSpanKind stronger_map_kind(TimelineSpanKind a, TimelineSpanKind b) {
  return map_kind_priority(b) > map_kind_priority(a) ? b : a;
}

ImU32 map_timeline_color(TimelineSpanKind kind, uint8_t alpha) {
  const TimelineColor color = timeline_span_color(kind, alpha);
  return IM_COL32(color.r, color.g, color.b, color.a);
}

struct BasemapRoadPaint {
  ImU32 casing = 0;
  ImU32 fill = 0;
  float casing_width = 1.0f;
  float fill_width = 1.0f;
};

BasemapRoadPaint basemap_road_paint(MapBasemapFeatureKind kind) {
  switch (kind) {
    case MapBasemapFeatureKind::RoadMotorway:
      return {
        .casing = ImGui::GetColorU32(color_rgb(92, 91, 86, 0.78f)),
        .fill = ImGui::GetColorU32(color_rgb(213, 199, 158, 0.92f)),
        .casing_width = 5.2f,
        .fill_width = 3.3f,
      };
    case MapBasemapFeatureKind::RoadPrimary:
      return {
        .casing = ImGui::GetColorU32(color_rgb(80, 84, 86, 0.74f)),
        .fill = ImGui::GetColorU32(color_rgb(202, 207, 202, 0.88f)),
        .casing_width = 4.2f,
        .fill_width = 2.6f,
      };
    case MapBasemapFeatureKind::RoadSecondary:
      return {
        .casing = ImGui::GetColorU32(color_rgb(74, 80, 82, 0.68f)),
        .fill = ImGui::GetColorU32(color_rgb(184, 193, 190, 0.82f)),
        .casing_width = 3.2f,
        .fill_width = 2.0f,
      };
    case MapBasemapFeatureKind::RoadLocal:
    default:
      return {
        .casing = ImGui::GetColorU32(color_rgb(68, 74, 77, 0.62f)),
        .fill = ImGui::GetColorU32(color_rgb(151, 162, 163, 0.72f)),
        .casing_width = 2.2f,
        .fill_width = 1.35f,
      };
  }
}

bool basemap_feature_intersects_projection(const MapBasemapFeature &feature, const MapProjection &projection) {
  if (!feature.bounds.valid()) return true;
  const double view_min_x = projection.min_x;
  const double view_max_x = projection.min_x + static_cast<double>(projection.size.x) / projection.scale;
  const double view_min_y = projection.min_y;
  const double view_max_y = projection.min_y + static_cast<double>(projection.size.y) / projection.scale;
  const double feature_min_x = projected_x(feature.bounds.west, projection.cos_lat);
  const double feature_max_x = projected_x(feature.bounds.east, projection.cos_lat);
  const double feature_min_y = projected_y(feature.bounds.north);
  const double feature_max_y = projected_y(feature.bounds.south);
  return !(feature_max_x < view_min_x || feature_min_x > view_max_x ||
           feature_max_y < view_min_y || feature_min_y > view_max_y);
}

std::vector<ImVec2> basemap_screen_points(const MapBasemapFeature &feature, const MapProjection &projection) {
  std::vector<ImVec2> points;
  points.reserve(feature.points.size());
  for (const MapBasemapPoint &point : feature.points) {
    points.push_back(map_to_screen(projection, point.lat, point.lon));
  }
  return points;
}

bool convex_screen_ring(const std::vector<ImVec2> &points) {
  if (points.size() < 4) return false;
  float sign = 0.0f;
  const size_t n = points.size();
  for (size_t i = 0; i < n; ++i) {
    const ImVec2 &a = points[i];
    const ImVec2 &b = points[(i + 1) % n];
    const ImVec2 &c = points[(i + 2) % n];
    const float cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x);
    if (std::abs(cross) < 1.0e-3f) continue;
    if (sign == 0.0f) {
      sign = cross;
    } else if ((cross > 0.0f) != (sign > 0.0f)) {
      return false;
    }
  }
  return sign != 0.0f;
}

void draw_basemap_polyline(ImDrawList *draw_list,
                           const std::vector<ImVec2> &points,
                           ImU32 color,
                           float thickness,
                           bool closed = false) {
  if (points.size() < 2) return;
  draw_list->AddPolyline(points.data(), static_cast<int>(points.size()), color,
                         closed ? ImDrawFlags_Closed : ImDrawFlags_None, thickness);
}

void draw_basemap_features(ImDrawList *draw_list, const MapBasemap &basemap, const MapProjection &projection) {
  const ImU32 water_fill = ImGui::GetColorU32(color_rgb(54, 93, 117, 0.42f));
  const ImU32 water_outline = ImGui::GetColorU32(color_rgb(88, 139, 168, 0.70f));
  const ImU32 water_line = ImGui::GetColorU32(color_rgb(96, 151, 181, 0.74f));

  for (const MapBasemapFeature &feature : basemap.features) {
    if (feature.kind != MapBasemapFeatureKind::WaterPolygon || !basemap_feature_intersects_projection(feature, projection)) continue;
    const std::vector<ImVec2> points = basemap_screen_points(feature, projection);
    if (points.size() >= 3 && convex_screen_ring(points)) {
      draw_list->AddConvexPolyFilled(points.data(), static_cast<int>(points.size()), water_fill);
    }
    draw_basemap_polyline(draw_list, points, water_outline, 1.7f, true);
  }

  for (const MapBasemapFeature &feature : basemap.features) {
    if (feature.kind != MapBasemapFeatureKind::WaterLine || !basemap_feature_intersects_projection(feature, projection)) continue;
    draw_basemap_polyline(draw_list, basemap_screen_points(feature, projection), water_line, 2.1f);
  }

  constexpr MapBasemapFeatureKind order[] = {
    MapBasemapFeatureKind::RoadLocal,
    MapBasemapFeatureKind::RoadSecondary,
    MapBasemapFeatureKind::RoadPrimary,
    MapBasemapFeatureKind::RoadMotorway,
  };
  for (MapBasemapFeatureKind kind : order) {
    const BasemapRoadPaint paint = basemap_road_paint(kind);
    for (const MapBasemapFeature &feature : basemap.features) {
      if (feature.kind != kind || !basemap_feature_intersects_projection(feature, projection)) continue;
      const std::vector<ImVec2> points = basemap_screen_points(feature, projection);
      draw_basemap_polyline(draw_list, points, paint.casing, paint.casing_width);
      draw_basemap_polyline(draw_list, points, paint.fill, paint.fill_width);
    }
  }
}

void clear_map_cache_directory(const fs::path &root) {
  std::error_code ec;
  if (!fs::exists(root, ec)) return;
  for (const fs::directory_entry &entry : fs::directory_iterator(root, ec)) {
    if (ec) break;
    if (!entry.is_regular_file(ec)) continue;
    fs::remove(entry.path(), ec);
  }
}

}  // namespace

MapBasemapBounds map_basemap_bounds_for_trace(const MapTrace &trace) {
  if (!trace.valid()) return {};
  const double center_lat = (trace.min_lat + trace.max_lat) * 0.5;
  const double lat_span = std::max(trace.max_lat - trace.min_lat, 0.002);
  const double lon_span = std::max(trace.max_lon - trace.min_lon, 0.002 / map_cos_lat_scale(center_lat));
  const double lat_pad = std::max(lat_span * kMapTracePadFrac, kMapTraceMinLatPad);
  const double lon_pad = std::max(lon_span * kMapTracePadFrac, kMapTraceMinLatPad / map_cos_lat_scale(center_lat));
  return {
    .south = clamp_basemap_lat(quantize_down(trace.min_lat - lat_pad, kMapBoundsGrid)),
    .west = clamp_basemap_lon(quantize_down(trace.min_lon - lon_pad, kMapBoundsGrid)),
    .north = clamp_basemap_lat(quantize_up(trace.max_lat + lat_pad, kMapBoundsGrid)),
    .east = clamp_basemap_lon(quantize_up(trace.max_lon + lon_pad, kMapBoundsGrid)),
  };
}

MapBasemapRequest map_basemap_request_for_trace(const MapTrace &trace) {
  if (!trace.valid()) return {};
  std::vector<MapBasemapBounds> boxes = corridor_boxes_for_trace(trace);
  if (boxes.empty()) boxes.push_back(map_basemap_bounds_for_trace(trace));

  MapBasemapBounds union_bounds;
  std::string query = "[out:json][timeout:25];(";
  for (const MapBasemapBounds &box : boxes) {
    union_bounds = merge_basemap_bounds(union_bounds, box);
    const std::string bbox = bbox_string(box);
    query += "way[\"highway\"][\"area\"!=\"yes\"](" + bbox + ");";
    query += "way[\"natural\"=\"water\"](" + bbox + ");";
    query += "way[\"waterway\"=\"riverbank\"](" + bbox + ");";
    query += "way[\"waterway\"~\"river|stream|canal\"](" + bbox + ");";
  }
  query += ");out tags geom;";

  std::string key = bounds_key(union_bounds);
  key += ":";
  key += std::to_string(boxes.size());
  for (const MapBasemapBounds &box : boxes) {
    key += ":";
    key += bbox_string(box);
  }

  return {
    .key = std::move(key),
    .bounds = union_bounds,
    .query = std::move(query),
  };
}

std::optional<MapBasemap> parse_map_basemap_json(std::string_view raw,
                                                 const MapBasemapBounds &bounds,
                                                 std::string key) {
  std::string parse_error;
  const json11::Json root = json11::Json::parse(std::string(raw), parse_error);
  if (!parse_error.empty() || !root.is_object() || !bounds.valid()) return std::nullopt;

  MapBasemap basemap;
  basemap.key = std::move(key);
  basemap.bounds = bounds;
  for (const json11::Json &element : root["elements"].array_items()) {
    if (element["type"].string_value() != "way") continue;
    const json11::Json &tags = element["tags"];
    std::vector<MapBasemapPoint> points = geometry_points(element["geometry"]);
    if (points.size() < 2) continue;

    const std::string highway = tags["highway"].string_value();
    if (!highway.empty()) {
      const std::optional<MapBasemapFeatureKind> kind = classify_road(highway);
      if (!kind.has_value()) continue;
      basemap.features.push_back({
        .kind = *kind,
        .bounds = feature_bounds(points),
        .points = std::move(points),
      });
      continue;
    }

    const std::string natural = tags["natural"].string_value();
    const std::string waterway = tags["waterway"].string_value();
    const bool closed = points.size() >= 4 &&
                        std::abs(points.front().lat - points.back().lat) < 1.0e-9 &&
                        std::abs(points.front().lon - points.back().lon) < 1.0e-9;
    if ((natural == "water" || waterway == "riverbank") && closed) {
      basemap.features.push_back({
        .kind = MapBasemapFeatureKind::WaterPolygon,
        .bounds = feature_bounds(points),
        .points = std::move(points),
      });
      continue;
    }
    if (waterway == "river" || waterway == "stream" || waterway == "canal") {
      basemap.features.push_back({
        .kind = MapBasemapFeatureKind::WaterLine,
        .bounds = feature_bounds(points),
        .points = std::move(points),
      });
    }
  }
  return basemap;
}

std::string map_basemap_cache_json(const MapBasemap &basemap) {
  json11::Json::array features;
  features.reserve(basemap.features.size());
  for (const MapBasemapFeature &feature : basemap.features) {
    json11::Json::array points;
    points.reserve(feature.points.size());
    for (const MapBasemapPoint &point : feature.points) {
      points.push_back(json11::Json::array{point.lat, point.lon});
    }
    features.push_back(json11::Json::object{
      {"kind", feature_kind_token(feature.kind)},
      {"bounds", bounds_json(feature.bounds)},
      {"points", points},
    });
  }
  return json11::Json(json11::Json::object{
    {"version", 1},
    {"key", basemap.key},
    {"bounds", bounds_json(basemap.bounds)},
    {"features", features},
  }).dump();
}

std::optional<MapBasemap> parse_map_basemap_cache_json(std::string_view raw,
                                                       std::string expected_key) {
  std::string parse_error;
  const json11::Json root = json11::Json::parse(std::string(raw), parse_error);
  if (!parse_error.empty() || !root.is_object() || root["version"].int_value() != 1) return std::nullopt;

  const std::string key = root["key"].string_value();
  if (key.empty() || (!expected_key.empty() && key != expected_key)) return std::nullopt;
  const std::optional<MapBasemapBounds> bounds = parse_bounds_json(root["bounds"]);
  if (!bounds.has_value()) return std::nullopt;

  MapBasemap basemap;
  basemap.key = key;
  basemap.bounds = *bounds;
  for (const json11::Json &item : root["features"].array_items()) {
    const std::optional<MapBasemapFeatureKind> kind = parse_feature_kind(item["kind"].string_value());
    if (!kind.has_value() || !item["points"].is_array()) continue;
    MapBasemapFeature feature;
    feature.kind = *kind;
    for (const json11::Json &point : item["points"].array_items()) {
      const auto values = point.array_items();
      if (values.size() != 2 || !values[0].is_number() || !values[1].is_number()) continue;
      const double lat = values[0].number_value();
      const double lon = values[1].number_value();
      if (valid_map_coordinate(lat, lon)) feature.points.push_back({.lat = lat, .lon = lon});
    }
    if (feature.points.size() < 2) continue;
    if (const std::optional<MapBasemapBounds> cached_bounds = parse_bounds_json(item["bounds"]); cached_bounds.has_value()) {
      feature.bounds = *cached_bounds;
    } else {
      feature.bounds = feature_bounds(feature.points);
    }
    basemap.features.push_back(std::move(feature));
  }
  return basemap;
}

std::filesystem::path default_map_basemap_cache_root() {
  if (const char *xdg = std::getenv("XDG_CACHE_HOME"); xdg != nullptr && xdg[0] != '\0') {
    return fs::path(xdg) / "loggy" / "map";
  }
  if (const char *home = std::getenv("HOME"); home != nullptr && home[0] != '\0') {
    return fs::path(home) / ".cache" / "loggy" / "map";
  }
  return fs::temp_directory_path() / "loggy" / "map";
}

std::filesystem::path map_basemap_effective_cache_root(std::string_view configured_root) {
  return configured_root.empty() ? default_map_basemap_cache_root()
                                 : fs::path(std::string(configured_root));
}

std::filesystem::path map_basemap_cache_path(const std::filesystem::path &root,
                                             std::string_view key) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%016llx.json", static_cast<unsigned long long>(fnv1a64(key)));
  return root / buf;
}

bool save_map_basemap_cache(const std::filesystem::path &path,
                            const MapBasemap &basemap,
                            std::string &error) {
  return write_text_file(path, map_basemap_cache_json(basemap), error);
}

std::optional<MapBasemap> load_map_basemap_cache(const std::filesystem::path &path,
                                                 std::string_view expected_key,
                                                 std::string &error) {
  const std::optional<std::string> raw = read_text_file(path, error);
  if (!raw.has_value()) return std::nullopt;
  return parse_map_basemap_cache_json(*raw, std::string(expected_key));
}

MapBasemapCacheStats map_basemap_cache_stats(const std::filesystem::path &root) {
  MapBasemapCacheStats stats;
  std::error_code ec;
  if (!fs::exists(root, ec)) return stats;
  for (const fs::directory_entry &entry : fs::directory_iterator(root, ec)) {
    if (ec) break;
    if (!entry.is_regular_file(ec)) continue;
    stats.bytes += static_cast<uint64_t>(entry.file_size(ec));
    ++stats.files;
  }
  return stats;
}

std::string map_basemap_cache_summary(const MapBasemapCacheStats &stats) {
  double value = static_cast<double>(stats.bytes);
  const char *unit = "B";
  if (value >= 1024.0) {
    value /= 1024.0;
    unit = "KB";
  }
  if (value >= 1024.0) {
    value /= 1024.0;
    unit = "MB";
  }

  char bytes_buf[64];
  if (unit[0] == 'B') {
    std::snprintf(bytes_buf, sizeof(bytes_buf), "%llu B",
                  static_cast<unsigned long long>(stats.bytes));
  } else {
    std::snprintf(bytes_buf, sizeof(bytes_buf), "%.1f %s", value, unit);
  }

  const char *file_label = stats.files == 1 ? "file" : "files";
  return std::to_string(stats.files) + " " + file_label + " / " + bytes_buf;
}

std::string map_basemap_cache_root_text(const fs::path &root) {
  return root.empty() ? std::string("(default)") : root.string();
}

std::string map_basemap_status_text(const MapBasemapStatus &status, const fs::path &root) {
  const std::string message = status.message.empty() ? "idle" : status.message;
  const std::string cache_root = map_basemap_cache_root_text(root);
  const std::string cache_summary = map_basemap_cache_summary(status.cache);
  std::string suffix = status.loading ? " | working" : "";
  if (status.fetching) suffix += " | fetching";
  if (!status.key.empty()) suffix += " | " + status.key;
  return message + suffix + " | root " + cache_root + " | cache " + cache_summary;
}

std::optional<MapBasemapPoint> map_coordinate_for_canvas_point(
  const MapTrace &trace, const MapState &state, double width, double height,
  double x, double y, const std::optional<MapTracePoint> &tracker) {
  if (!trace.valid() || width <= 0.0 || height <= 0.0 ||
      !std::isfinite(x) || !std::isfinite(y)) {
    return std::nullopt;
  }
  const MapProjection projection = make_projection(
    trace, ImVec2(0.0f, 0.0f), ImVec2(static_cast<float>(width), static_cast<float>(height)), state, tracker);
  const double clamped_x = std::clamp(x, 0.0, width);
  const double clamped_y = std::clamp(y, 0.0, height);
  const double projected_anchor_x = projection.min_x + clamped_x / projection.scale;
  const double projected_anchor_y = projection.min_y + clamped_y / projection.scale;
  return MapBasemapPoint{
    .lat = std::clamp(-projected_anchor_y, -90.0, 90.0),
    .lon = std::clamp(projected_anchor_x / std::max(0.20, projection.cos_lat), -180.0, 180.0),
  };
}

bool map_zoom_about_canvas_point(MapState *state, const MapTrace &trace,
                                 double width, double height, double x, double y,
                                 double zoom_multiplier,
                                 const std::optional<MapTracePoint> &tracker) {
  if (state == nullptr || !trace.valid() || width <= 0.0 || height <= 0.0 ||
      !std::isfinite(x) || !std::isfinite(y) ||
      !std::isfinite(zoom_multiplier) || zoom_multiplier <= 0.0) {
    return false;
  }

  const double old_zoom = std::clamp(state->zoom, kMinMapZoom, kMaxMapZoom);
  const double new_zoom = std::clamp(old_zoom * zoom_multiplier, kMinMapZoom, kMaxMapZoom);
  if (std::abs(new_zoom - old_zoom) < 1.0e-12) return false;

  const ImVec2 size(static_cast<float>(width), static_cast<float>(height));
  const MapProjection before = make_projection(trace, ImVec2(0.0f, 0.0f), size, *state, tracker);
  const double clamped_x = std::clamp(x, 0.0, width);
  const double clamped_y = std::clamp(y, 0.0, height);
  const double anchor_x = before.min_x + clamped_x / before.scale;
  const double anchor_y = before.min_y + clamped_y / before.scale;

  set_map_center_from_projection(state, before);
  state->follow = false;
  state->zoom = new_zoom;

  const MapProjection after_zoom = make_projection(trace, ImVec2(0.0f, 0.0f), size, *state, tracker);
  const double center_x = anchor_x - clamped_x / after_zoom.scale + width * 0.5 / after_zoom.scale;
  const double center_y = anchor_y - clamped_y / after_zoom.scale + height * 0.5 / after_zoom.scale;
  set_map_center_from_projected(state, center_x, center_y, after_zoom.cos_lat);
  return true;
}

bool map_pan_by_canvas_delta(MapState *state, const MapTrace &trace,
                             double width, double height, double delta_x, double delta_y,
                             bool natural_drag,
                             const std::optional<MapTracePoint> &tracker) {
  if (state == nullptr || !trace.valid() || width <= 0.0 || height <= 0.0 ||
      !std::isfinite(delta_x) || !std::isfinite(delta_y) ||
      (std::abs(delta_x) < 1.0e-12 && std::abs(delta_y) < 1.0e-12)) {
    return false;
  }

  const ImVec2 size(static_cast<float>(width), static_cast<float>(height));
  const MapProjection projection = make_projection(trace, ImVec2(0.0f, 0.0f), size, *state, tracker);
  set_map_center_from_projection(state, projection);
  state->follow = false;

  const double direction = natural_drag ? -1.0 : 1.0;
  const double center_x = projection.min_x + static_cast<double>(projection.size.x) * 0.5 / projection.scale +
                          direction * delta_x / projection.scale;
  const double center_y = projection.min_y + static_cast<double>(projection.size.y) * 0.5 / projection.scale +
                          direction * delta_y / projection.scale;
  set_map_center_from_projected(state, center_x, center_y, projection.cos_lat);
  return true;
}

bool map_basemap_matches_trace(const MapTrace &trace, const MapBasemap *basemap) {
  if (!trace.valid() || basemap == nullptr) return false;
  const MapBasemapRequest request = map_basemap_request_for_trace(trace);
  return !request.key.empty() && basemap->key == request.key;
}

struct MapBasemapManager {
  struct Request {
    MapBasemapRequest spec;
    fs::path cache_root;
    uint64_t generation = 0;
    bool allow_fetch = false;
  };

  MapBasemapManager() : worker([this]() { run(); }) {
    snapshot.cache = map_basemap_cache_stats(cache_root);
    snapshot.message = "No basemap loaded";
  }

  ~MapBasemapManager() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      stopping = true;
    }
    cv.notify_all();
    if (worker.joinable()) worker.join();
  }

  void pump() {
    std::unique_ptr<MapBasemap> ready;
    {
      std::lock_guard<std::mutex> lock(mutex);
      ready = std::move(completed);
    }
    if (ready) current = std::move(ready);
  }

  void ensureTrace(const MapTrace &trace, bool allow_fetch) {
    const MapBasemapRequest spec = map_basemap_request_for_trace(trace);
    if (!spec.bounds.valid() || spec.key.empty()) return;
    if (current && current->key == spec.key) return;

    std::lock_guard<std::mutex> lock(mutex);
    if (!allow_fetch && last_miss_key == spec.key) return;
    if (pending && pending->spec.key == spec.key && pending->generation == generation) {
      pending->allow_fetch = pending->allow_fetch || allow_fetch;
      return;
    }
    if (active && active->spec.key == spec.key && active->generation == generation) return;
    pending = Request{
      .spec = spec,
      .cache_root = cache_root,
      .generation = generation,
      .allow_fetch = allow_fetch,
    };
    snapshot.loading = true;
    snapshot.fetching = false;
    snapshot.key = spec.key;
    snapshot.message = allow_fetch ? "Queued basemap fetch" : "Queued basemap cache check";
    cv.notify_one();
  }

  void setCacheRoot(fs::path root) {
    if (root.empty()) root = default_map_basemap_cache_root();
    current.reset();
    std::lock_guard<std::mutex> lock(mutex);
    if (root == cache_root) return;
    cache_root = std::move(root);
    ++generation;
    pending.reset();
    active.reset();
    completed.reset();
    last_miss_key.clear();
    refresh_requested = true;
    snapshot.loading = true;
    snapshot.fetching = false;
    snapshot.key.clear();
    snapshot.message = "Refreshing map cache";
    cv.notify_one();
  }

  fs::path cacheRoot() const {
    std::lock_guard<std::mutex> lock(mutex);
    return cache_root;
  }

  void clearCache() {
    current.reset();
    std::lock_guard<std::mutex> lock(mutex);
    last_miss_key.clear();
    clear_requested = true;
    snapshot.loading = true;
    snapshot.fetching = false;
    snapshot.message = "Clearing map cache";
    cv.notify_one();
  }

  void refreshCacheStats() {
    std::lock_guard<std::mutex> lock(mutex);
    refresh_requested = true;
    snapshot.loading = true;
    snapshot.fetching = false;
    snapshot.message = "Refreshing map cache";
    cv.notify_one();
  }

  MapBasemapStatus status() const {
    std::lock_guard<std::mutex> lock(mutex);
    return snapshot;
  }

  void setStatusLocked(std::string message, bool loading, bool fetching) {
    snapshot.message = std::move(message);
    snapshot.loading = loading;
    snapshot.fetching = fetching;
  }

  void run() {
    while (true) {
      Request request;
      bool should_clear = false;
      bool should_refresh = false;
      {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&]() {
          return stopping || clear_requested || refresh_requested || pending.has_value();
        });
        if (stopping) return;
        if (clear_requested) {
          clear_requested = false;
          should_clear = true;
        } else if (refresh_requested) {
          refresh_requested = false;
          should_refresh = true;
        } else {
          request = *pending;
          active = std::move(*pending);
          pending.reset();
          setStatusLocked(request.allow_fetch ? "Fetching basemap" : "Checking basemap cache",
                          true, request.allow_fetch);
        }
      }

      if (should_clear) {
        fs::path root;
        {
          std::lock_guard<std::mutex> lock(mutex);
          root = cache_root;
        }
        clear_map_cache_directory(root);
        const MapBasemapCacheStats stats = map_basemap_cache_stats(root);
        std::lock_guard<std::mutex> lock(mutex);
        setStatusLocked("Map cache cleared", false, false);
        snapshot.cache = stats;
        continue;
      }

      if (should_refresh) {
        fs::path root;
        {
          std::lock_guard<std::mutex> lock(mutex);
          root = cache_root;
        }
        const MapBasemapCacheStats stats = map_basemap_cache_stats(root);
        std::lock_guard<std::mutex> lock(mutex);
        setStatusLocked("Map cache refreshed", false, false);
        snapshot.cache = stats;
        continue;
      }

      std::unique_ptr<MapBasemap> parsed;
      std::string message = "No basemap cache";
      std::string error;
      const fs::path cache_path = map_basemap_cache_path(request.cache_root, request.spec.key);
      if (auto cached = load_map_basemap_cache(cache_path, request.spec.key, error)) {
        parsed = std::make_unique<MapBasemap>(std::move(*cached));
        message = "Loaded basemap cache";
      } else if (request.allow_fetch) {
        std::string raw;
        if (auto fetched = fetch_overpass_json(request.spec.query)) {
          raw = std::move(*fetched);
          if (auto basemap = parse_map_basemap_json(raw, request.spec.bounds, request.spec.key)) {
            std::string save_error;
            save_map_basemap_cache(cache_path, *basemap, save_error);
            parsed = std::make_unique<MapBasemap>(std::move(*basemap));
            message = "Fetched basemap";
            if (!save_error.empty()) {
              message = "Fetched basemap (cache write failed: " + save_error + ")";
            }
          } else {
            message = "Basemap parse failed";
          }
        } else {
          message = "Basemap fetch failed";
        }
      } else if (!error.empty()) {
        message = "Basemap cache failed: " + error;
      }
      if (message.find("failed") == std::string::npos && !error.empty() && parsed == nullptr) {
        message = "Basemap cache failed: " + error;
      }

      const MapBasemapCacheStats stats = map_basemap_cache_stats(request.cache_root);
      {
        std::lock_guard<std::mutex> lock(mutex);
        if (active && active->spec.key == request.spec.key &&
            active->generation == request.generation && generation == request.generation) {
          completed = std::move(parsed);
          if (completed) {
            last_miss_key.clear();
          } else {
            last_miss_key = request.spec.key;
          }
          active.reset();
          setStatusLocked(message, false, false);
          snapshot.cache = stats;
        }
      }
    }
  }

  mutable std::mutex mutex;
  std::condition_variable cv;
  bool stopping = false;
  bool clear_requested = false;
  bool refresh_requested = false;
  fs::path cache_root = default_map_basemap_cache_root();
  uint64_t generation = 0;
  std::optional<Request> pending;
  std::optional<Request> active;
  std::unique_ptr<MapBasemap> completed;
  std::unique_ptr<MapBasemap> current;
  std::string last_miss_key;
  MapBasemapStatus snapshot;
  std::thread worker;
};

MapBasemapManager &map_basemap_manager(PaneInstance &pane) {
  MapPaneTransientState &transient = map_pane_transient_state(pane);
  if (!transient.basemap) transient.basemap = std::make_shared<MapBasemapManager>();
  return *transient.basemap;
}

void draw_map_pane(Session &session, PaneInstance &pane) {
  MapState &state = map_pane_state(pane);
  MapBasemapManager &basemap_manager = map_basemap_manager(pane);
  basemap_manager.setCacheRoot(map_basemap_effective_cache_root(session.settings.map_cache_root));
  TimeRange range = session.playback.route_range();
  if (!range.valid() || range.span() <= 0.0) range = session.view_range.range();
  const MapTrace trace = prepare_map_trace(session.store, range, state, &session.timeline);
  const std::optional<MapTracePoint> tracker = map_trace_point_at_time(trace, session.playback.tracker_time());
  if (trace.valid() && state.show_basemap) {
    basemap_manager.pump();
    basemap_manager.ensureTrace(trace, false);
  }
  const MapBasemapStatus basemap_status = basemap_manager.status();
  const MapBasemap *loaded_basemap = state.show_basemap ? basemap_manager.current.get() : nullptr;
  const bool loaded_basemap_matches = map_basemap_matches_trace(trace, loaded_basemap);
  const MapBasemap *basemap = loaded_basemap_matches ? loaded_basemap : nullptr;
  bool changed = false;

  ImGui::TextDisabled("%zu GPS points", trace.points.size());
  if (trace.valid()) {
    ImGui::SameLine();
    ImGui::TextDisabled("| %.5f, %.5f to %.5f, %.5f", trace.min_lat, trace.min_lon, trace.max_lat, trace.max_lon);
    if (trace.decimated) {
      ImGui::SameLine();
      ImGui::TextDisabled("| decimated");
    }
  }
  changed = ImGui::Checkbox("Follow", &state.follow) || changed;
  ImGui::SameLine();
  changed = ImGui::Checkbox("Basemap", &state.show_basemap) || changed;
  ImGui::SameLine();
  if (ImGui::Button("-")) {
    state.zoom = std::clamp(state.zoom / 1.25, kMinMapZoom, kMaxMapZoom);
    changed = true;
  }
  ImGui::SameLine();
  ImGui::TextDisabled("%.2gx", state.zoom);
  ImGui::SameLine();
  if (ImGui::Button("+")) {
    state.zoom = std::clamp(state.zoom * 1.25, kMinMapZoom, kMaxMapZoom);
    changed = true;
  }
  ImGui::SameLine();
  if (ImGui::Button("Fit")) {
    state.zoom = 1.0;
    state.follow = false;
    state.has_center = false;
    changed = true;
  }
  const std::string maps_url = map_google_maps_url(trace);
  if (!maps_url.empty()) {
    ImGui::SameLine();
    if (ImGui::Button("Maps")) {
      ImGui::SetClipboardText(maps_url.c_str());
    }
  }
  if (trace.valid() && state.show_basemap) {
    if (ImGui::SmallButton("Fetch")) {
      basemap_manager.ensureTrace(trace, true);
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Fetch Overpass roads and water for this GPS trace");
    ImGui::SameLine();
    if (ImGui::SmallButton("Clear")) {
      basemap_manager.clearCache();
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Clear cached map features");
    ImGui::SameLine();
    if (ImGui::SmallButton("Cache")) {
      ImGui::OpenPopup("##map_cache_popup");
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Manage map cache");
    if (ImGui::BeginPopup("##map_cache_popup")) {
      const fs::path cache_root = basemap_manager.cacheRoot();
      const std::string cache_root_text = map_basemap_cache_root_text(cache_root);
      ImGui::TextUnformatted("Map Cache");
      ImGui::TextDisabled("%s", cache_root_text.c_str());
      ImGui::TextDisabled("%s", map_basemap_cache_summary(basemap_status.cache).c_str());
      if (!basemap_status.key.empty()) {
        ImGui::TextDisabled("Key: %.48s%s",
                            basemap_status.key.c_str(),
                            basemap_status.key.size() > 48 ? "..." : "");
      }
      ImGui::Separator();
      if (ImGui::Button("Copy Path", ImVec2(92.0f, 0.0f))) {
        ImGui::SetClipboardText(cache_root_text.c_str());
      }
      ImGui::SameLine();
      if (ImGui::Button("Refresh", ImVec2(92.0f, 0.0f))) {
        basemap_manager.refreshCacheStats();
      }
      ImGui::SameLine();
      if (ImGui::Button("Clear", ImVec2(92.0f, 0.0f))) {
        basemap_manager.clearCache();
      }
      ImGui::EndPopup();
    }
    const char *status_message = basemap_status.message.empty() ? "idle" : basemap_status.message.c_str();
    const char *match_note = loaded_basemap != nullptr && !loaded_basemap_matches ? " | refreshing" : "";
    ImGui::TextDisabled("%s%s | %zu feat | %s",
                        status_message,
                        basemap_status.loading ? "..." : match_note,
                        basemap != nullptr ? basemap->features.size() : 0,
                        map_basemap_cache_summary(basemap_status.cache).c_str());
  }

  const ImVec2 avail = ImGui::GetContentRegionAvail();
  const ImVec2 size(std::max(1.0f, avail.x), std::max(1.0f, avail.y));
  const ImVec2 origin = ImGui::GetCursorScreenPos();
  ImGui::InvisibleButton("##map_canvas", size, ImGuiButtonFlags_MouseButtonLeft);
  const bool canvas_hovered = ImGui::IsItemHovered();
  const bool canvas_active = ImGui::IsItemActive();
  ImDrawList *draw_list = ImGui::GetWindowDrawList();
  draw_map_grid(draw_list, origin, size);

  if (!trace.valid()) {
    const char *label = "No GPS trace in store";
    const ImVec2 text_size = ImGui::CalcTextSize(label);
    draw_list->AddText(ImVec2(origin.x + (size.x - text_size.x) * 0.5f,
                              origin.y + (size.y - text_size.y) * 0.5f),
                       ImGui::GetColorU32(ImGuiCol_TextDisabled), label);
    if (changed) save_map_pane_state(pane, state);
    return;
  }

  MapProjection projection = make_projection(trace, origin, size, state, tracker);
  if (canvas_hovered && ImGui::GetIO().MouseWheel != 0.0f) {
    const ImVec2 mouse = ImGui::GetIO().MousePos;
    const double zoom_multiplier = std::pow(1.15, static_cast<double>(ImGui::GetIO().MouseWheel));
    if (map_zoom_about_canvas_point(&state, trace, size.x, size.y,
                                    mouse.x - origin.x, mouse.y - origin.y,
                                    zoom_multiplier, tracker)) {
      changed = true;
      projection = make_projection(trace, origin, size, state, tracker);
    }
  }
  if (canvas_active && ImGui::IsMouseDragging(ImGuiMouseButton_Left, 0.0f)) {
    const ImVec2 delta = ImGui::GetIO().MouseDelta;
    if (map_pan_by_canvas_delta(&state, trace, size.x, size.y,
                                delta.x, delta.y,
                                session.settings.natural_map_drag, tracker)) {
      changed = true;
      projection = make_projection(trace, origin, size, state, tracker);
    }
  }

  if (state.show_basemap && basemap != nullptr) {
    draw_basemap_features(draw_list, *basemap, projection);
  }

  std::vector<ImVec2> points;
  points.reserve(trace.points.size());
  for (const MapTracePoint &point : trace.points) {
    points.push_back(map_to_screen(projection, point.lat, point.lon));
  }
  if (points.size() >= 2) {
    for (size_t i = 1; i < points.size(); ++i) {
      const TimelineSpanKind kind = stronger_map_kind(trace.points[i - 1].kind, trace.points[i].kind);
      draw_list->AddLine(points[i - 1], points[i], map_timeline_color(kind, 88), 5.5f);
    }
    for (size_t i = 1; i < points.size(); ++i) {
      const TimelineSpanKind kind = stronger_map_kind(trace.points[i - 1].kind, trace.points[i].kind);
      draw_list->AddLine(points[i - 1], points[i], map_timeline_color(kind, 255), 2.2f);
    }
  } else if (points.size() == 1) {
    draw_list->AddCircleFilled(points.front(), 4.0f, map_timeline_color(trace.points.front().kind, 255));
  }

  if (tracker.has_value()) {
    const ImVec2 marker = map_to_screen(projection, tracker->lat, tracker->lon);
    draw_car_marker(draw_list, marker, tracker->bearing_deg, ImGui::GetColorU32(color_rgb(238, 188, 82)));
  }

  if (changed) save_map_pane_state(pane, state);
}

}  // namespace loggy
