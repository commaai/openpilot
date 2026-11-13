// Adapted from tools/jotpluggler/map.cc; preserve its basemap cache and interaction behavior.
#define IMGUI_DEFINE_MATH_OPERATORS
#include "tools/cabana/ui/chart/mapwidget.h"
#include "tools/cabana/ui/icons.h"
#include "imgui.h"
#include "imgui_internal.h"
#include "common/util.h"
#include "json11/json11.hpp"
#include <GLFW/glfw3.h>
#include "tools/replay/py_process.h"
#include <zstd.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <functional>
#include <limits>
#include <mutex>
#include <optional>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>
namespace cabana {
namespace map_detail {
namespace fs = std::filesystem;
struct GpsTrace;
struct GeoBounds {
  double south = 0.0;
  double west = 0.0;
  double north = 0.0;
  double east = 0.0;

  bool valid() const {
    return south < north && west < east;
  }
};

struct RouteBasemap;
struct MapCacheStats {
  uint64_t bytes = 0;
  size_t files = 0;
};

class MapDataManager {
public:
  MapDataManager();
  ~MapDataManager();

  MapDataManager(const MapDataManager &) = delete;
  MapDataManager &operator=(const MapDataManager &) = delete;

  void pump();
  void ensureTrace(const GpsTrace &trace);
  void clearCache();
  bool loading() const;
  std::string error() const;
  const RouteBasemap *current() const;
  MapCacheStats cacheStats() const;

private:
  struct Request {
    std::string key;
    GeoBounds bounds;
    std::string query;
  };

  void run();

  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::atomic<bool> stopping_{false};
  std::unique_ptr<Request> pending_;
  std::unique_ptr<Request> active_;
  std::unique_ptr<RouteBasemap> completed_;
  std::unique_ptr<RouteBasemap> current_;
  std::thread worker_;
  std::string error_, failed_key_;
  std::chrono::steady_clock::time_point retry_after_{};
};

struct TimelineEntry { enum class Type : uint8_t { None, Engaged, AlertInfo, AlertWarning, AlertCritical }; };
struct GpsPoint { double time = 0, lat = 0, lon = 0; float bearing = 0; TimelineEntry::Type type = TimelineEntry::Type::None; };
struct GpsTrace { std::vector<GpsPoint> points; double min_lat = 0, max_lat = 0, min_lon = 0, max_lon = 0; };
enum class SessionDataMode { Route, Stream };
struct TabUiState {
  struct MapPaneState { bool initialized = false, follow = false; float zoom = 1; double center_lat = 0, center_lon = 0; };
  std::vector<MapPaneState> map_panes{1};
};
struct UiState { TabUiState tab; bool has_tracker_time = true; double tracker_time = 0; std::string status_text; };
TabUiState *app_active_tab_state(UiState *state) { return &state->tab; }
struct AppSession { struct { GpsTrace gps_trace; } route_data; std::shared_ptr<MapDataManager> map_data;
  bool async_route_loading = false; SessionDataMode data_mode = SessionDataMode::Route; };
static void ensure_parent_dir(const std::filesystem::path &path) { std::filesystem::create_directories(path.parent_path()); }
static std::string route_google_maps_url(const GpsTrace &trace) {
  if (trace.points.empty()) return {};
  const auto &point = trace.points.front();
  return util::string_format("https://www.google.com/maps/search/?api=1&query=%.7f,%.7f", point.lat, point.lon);
}
static void open_external_url(const std::string &url) {
  // The URL is generated exclusively from numeric coordinates.
#ifdef __APPLE__
  const std::string command = "open '" + url + "' >/dev/null 2>&1 &";
#else
  const std::string command = "xdg-open '" + url + "' >/dev/null 2>&1 &";
#endif
  const int status = std::system(command.c_str());
  (void)status;
}
static ImU32 timeline_entry_color(TimelineEntry::Type type, float alpha, std::array<int, 3> fallback) {
  const std::array<ImU32, 5> colors{IM_COL32(fallback[0], fallback[1], fallback[2], 255), IM_COL32(0, 170, 100, 255),
    IM_COL32(50, 130, 220, 255), IM_COL32(240, 170, 0, 255), IM_COL32(220, 40, 40, 255)};
  const ImU32 value = colors[static_cast<size_t>(type)];
  return (value & 0x00ffffff) | (static_cast<ImU32>(std::clamp(alpha, 0.0f, 1.0f) * 255) << 24);
}


namespace {

constexpr int MAP_MIN_ZOOM = 1;
constexpr int MAP_MAX_ZOOM = 18;
constexpr int MAP_SINGLE_POINT_MIN_ZOOM = 14;
constexpr float MAP_WHEEL_ZOOM_STEP = 0.25f;
constexpr double MAP_TRACE_PAD_FRAC = 0.45;
constexpr double MAP_TRACE_MIN_LAT_PAD = 0.01;
constexpr double MAP_BOUNDS_GRID = 0.005;
constexpr double MAP_CORRIDOR_LAT_PAD = 0.010;
constexpr double MAP_CORRIDOR_MIN_STEP_S = 1.5;
constexpr size_t MAP_CORRIDOR_MAX_BOXES = 36;
constexpr float MAP_INITIAL_FIT_FILL = 0.88f;
constexpr float MAP_MIN_ZOOM_FILL = 0.98f;
constexpr float MAP_EDGE_FADE_FRAC = 0.28f;
constexpr const char *MAP_QUERY_ENDPOINTS[] = {
  "https://overpass-api.de/api/interpreter",
  "https://overpass.private.coffee/api/interpreter",
};
struct GeoPoint {
  double lat = 0.0;
  double lon = 0.0;
};

struct ProjectedPoint {
  float x = 0.0f;
  float y = 0.0f;
};

struct ProjectedBounds {
  float min_x = 0.0f;
  float min_y = 0.0f;
  float max_x = 0.0f;
  float max_y = 0.0f;

  bool valid() const {
    return max_x >= min_x && max_y >= min_y;
  }
};

enum class RoadClass : uint8_t {
  Motorway,
  Primary,
  Secondary,
  Local,
};

struct RoadFeature {
  RoadClass road_class = RoadClass::Local;
  ProjectedBounds bounds;
  std::vector<ProjectedPoint> points;
};

struct WaterLineFeature {
  ProjectedBounds bounds;
  std::vector<ProjectedPoint> points;
};

struct WaterPolygonFeature {
  ProjectedBounds bounds;
  std::vector<ProjectedPoint> ring;
};

}  // namespace

struct RouteBasemap {
  std::string key;
  GeoBounds bounds;
  ProjectedBounds projected_bounds;
  std::vector<RoadFeature> roads;
  std::vector<WaterLineFeature> water_lines;
  std::vector<WaterPolygonFeature> water_polygons;
};

struct MapRequestSpec {
  std::string key;
  GeoBounds bounds;
  std::string query;
};

namespace {

double lon_to_world_x(double lon, double zoom) {
  return (lon + 180.0) / 360.0 * 256.0 * std::exp2(zoom);
}

double lat_to_world_y(double lat, double zoom) {
  const double lat_rad = lat * M_PI / 180.0;
  return (1.0 - std::log(std::tan(lat_rad) + 1.0 / std::cos(lat_rad)) / M_PI) / 2.0 * 256.0 * std::exp2(zoom);
}

double world_x_to_lon(double x, double zoom) {
  return x / std::exp2(zoom) / 256.0 * 360.0 - 180.0;
}

double world_y_to_lat(double y, double zoom) {
  const double n = M_PI - (2.0 * M_PI * (y / std::exp2(zoom))) / 256.0;
  return 180.0 / M_PI * std::atan(std::sinh(n));
}

double map_trace_center_lat(const GpsTrace &trace) {
  return (trace.min_lat + trace.max_lat) * 0.5;
}

double map_trace_center_lon(const GpsTrace &trace) {
  return (trace.min_lon + trace.max_lon) * 0.5;
}

double clamp_lat(double lat) {
  return std::clamp(lat, -85.0, 85.0);
}

double clamp_lon(double lon) {
  return std::clamp(lon, -179.999, 179.999);
}

float project_lon0(double lon) {
  return static_cast<float>((lon + 180.0) / 360.0 * 256.0);
}

float project_lat0(double lat) {
  const double lat_rad = lat * M_PI / 180.0;
  return static_cast<float>((1.0 - std::log(std::tan(lat_rad) + 1.0 / std::cos(lat_rad)) / M_PI) / 2.0 * 256.0);
}

double cos_lat_scale(double lat) {
  return std::max(0.2, std::cos(lat * M_PI / 180.0));
}

double quantize_down(double value, double step) {
  return std::floor(value / step) * step;
}

double quantize_up(double value, double step) {
  return std::ceil(value / step) * step;
}

ProjectedBounds compute_projected_bounds(const std::vector<ProjectedPoint> &points) {
  ProjectedBounds bounds;
  if (points.empty()) {
    return bounds;
  }
  bounds.min_x = bounds.max_x = points.front().x;
  bounds.min_y = bounds.max_y = points.front().y;
  for (const ProjectedPoint &point : points) {
    bounds.min_x = std::min(bounds.min_x, point.x);
    bounds.max_x = std::max(bounds.max_x, point.x);
    bounds.min_y = std::min(bounds.min_y, point.y);
    bounds.max_y = std::max(bounds.max_y, point.y);
  }
  return bounds;
}

ProjectedBounds project_bounds0(const GeoBounds &bounds) {
  if (!bounds.valid()) {
    return {};
  }
  return ProjectedBounds{
    .min_x = project_lon0(bounds.west),
    .min_y = project_lat0(bounds.north),
    .max_x = project_lon0(bounds.east),
    .max_y = project_lat0(bounds.south),
  };
}

bool feature_intersects_view(const ProjectedBounds &feature, const ProjectedBounds &view, float zoom_scale) {
  const float min_x = feature.min_x * zoom_scale;
  const float max_x = feature.max_x * zoom_scale;
  const float min_y = feature.min_y * zoom_scale;
  const float max_y = feature.max_y * zoom_scale;
  return !(max_x < view.min_x || min_x > view.max_x
        || max_y < view.min_y || min_y > view.max_y);
}

GeoBounds requested_bounds_for_trace(const GpsTrace &trace) {
  if (trace.points.empty()) {
    return {};
  }
  const double center_lat = map_trace_center_lat(trace);
  const double lat_span = std::max(trace.max_lat - trace.min_lat, 0.002);
  const double lon_span = std::max(trace.max_lon - trace.min_lon, 0.002 / cos_lat_scale(center_lat));
  const double lat_pad = std::max(lat_span * MAP_TRACE_PAD_FRAC, MAP_TRACE_MIN_LAT_PAD);
  const double lon_pad = std::max(lon_span * MAP_TRACE_PAD_FRAC, MAP_TRACE_MIN_LAT_PAD / cos_lat_scale(center_lat));

  GeoBounds bounds;
  bounds.south = clamp_lat(quantize_down(trace.min_lat - lat_pad, MAP_BOUNDS_GRID));
  bounds.north = clamp_lat(quantize_up(trace.max_lat + lat_pad, MAP_BOUNDS_GRID));
  bounds.west = clamp_lon(quantize_down(trace.min_lon - lon_pad, MAP_BOUNDS_GRID));
  bounds.east = clamp_lon(quantize_up(trace.max_lon + lon_pad, MAP_BOUNDS_GRID));
  return bounds;
}

GeoBounds merge_bounds(const GeoBounds &a, const GeoBounds &b) {
  if (!a.valid()) return b;
  if (!b.valid()) return a;
  return GeoBounds{
    .south = std::min(a.south, b.south),
    .west = std::min(a.west, b.west),
    .north = std::max(a.north, b.north),
    .east = std::max(a.east, b.east),
  };
}

bool bounds_overlap_or_touch(const GeoBounds &a, const GeoBounds &b) {
  return !(a.east < b.west || b.east < a.west || a.north < b.south || b.north < a.south);
}

std::vector<GeoBounds> corridor_boxes_for_trace(const GpsTrace &trace) {
  std::vector<GeoBounds> boxes;
  if (trace.points.empty()) {
    return boxes;
  }

  const double center_lat = map_trace_center_lat(trace);
  const double lon_pad = MAP_CORRIDOR_LAT_PAD / cos_lat_scale(center_lat);
  const double total_time = trace.points.back().time - trace.points.front().time;
  const double target_boxes = std::min<double>(MAP_CORRIDOR_MAX_BOXES, std::max<double>(8.0, total_time / MAP_CORRIDOR_MIN_STEP_S));
  const size_t stride = std::max<size_t>(1, static_cast<size_t>(std::ceil(trace.points.size() / target_boxes)));

  auto add_box = [&](double lat, double lon) {
    GeoBounds box{
      .south = clamp_lat(quantize_down(lat - MAP_CORRIDOR_LAT_PAD, MAP_BOUNDS_GRID)),
      .west = clamp_lon(quantize_down(lon - lon_pad, MAP_BOUNDS_GRID)),
      .north = clamp_lat(quantize_up(lat + MAP_CORRIDOR_LAT_PAD, MAP_BOUNDS_GRID)),
      .east = clamp_lon(quantize_up(lon + lon_pad, MAP_BOUNDS_GRID)),
    };
    if (!box.valid()) {
      return;
    }
    for (GeoBounds &existing : boxes) {
      if (bounds_overlap_or_touch(existing, box)) {
        existing = merge_bounds(existing, box);
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
        if (bounds_overlap_or_touch(boxes[i], boxes[j])) {
          boxes[i] = merge_bounds(boxes[i], boxes[j]);
          boxes.erase(boxes.begin() + static_cast<std::ptrdiff_t>(j));
          merged = true;
          break;
        }
      }
    }
  }
  return boxes;
}

ProjectedBounds view_bounds(double top_left_x, double top_left_y, float width, float height) {
  return ProjectedBounds{
    .min_x = static_cast<float>(top_left_x),
    .min_y = static_cast<float>(top_left_y),
    .max_x = static_cast<float>(top_left_x + width),
    .max_y = static_cast<float>(top_left_y + height),
  };
}

int fit_map_zoom_for_bounds(const GeoBounds &bounds, float width, float height, float fill_fraction) {
  if (!bounds.valid()) {
    return MAP_MIN_ZOOM;
  }
  const double max_width = std::max(1.0f, width * fill_fraction);
  const double max_height = std::max(1.0f, height * fill_fraction);
  for (int z = MAP_MAX_ZOOM; z >= MAP_MIN_ZOOM; --z) {
    const double pixel_width = std::abs(lon_to_world_x(bounds.east, z) - lon_to_world_x(bounds.west, z));
    const double pixel_height = std::abs(lat_to_world_y(bounds.south, z) - lat_to_world_y(bounds.north, z));
    if (pixel_width <= max_width && pixel_height <= max_height) {
      return z;
    }
  }
  return MAP_MIN_ZOOM;
}

int fit_map_zoom_for_trace(const GpsTrace &trace, float width, float height) {
  return fit_map_zoom_for_bounds(requested_bounds_for_trace(trace), width, height, MAP_INITIAL_FIT_FILL);
}

int minimum_allowed_map_zoom(const GeoBounds &bounds, const GpsTrace &trace, ImVec2 size) {
  if (trace.points.size() <= 1) {
    return MAP_SINGLE_POINT_MIN_ZOOM;
  }
  const int fit_zoom = fit_map_zoom_for_bounds(bounds.valid() ? bounds : requested_bounds_for_trace(trace),
                                               size.x, size.y, MAP_MIN_ZOOM_FILL);
  return std::clamp(fit_zoom, MAP_MIN_ZOOM, MAP_MAX_ZOOM);
}

std::optional<GpsPoint> interpolate_gps(const GpsTrace &trace, double time_value) {
  if (trace.points.empty()) {
    return std::nullopt;
  }
  if (time_value <= trace.points.front().time) {
    return trace.points.front();
  }
  if (time_value >= trace.points.back().time) {
    return trace.points.back();
  }
  auto upper = std::lower_bound(trace.points.begin(), trace.points.end(), time_value,
                                [](const GpsPoint &point, double target) {
                                  return point.time < target;
                                });
  if (upper == trace.points.begin()) {
    return trace.points.front();
  }
  const GpsPoint &p1 = *upper;
  const GpsPoint &p0 = *(upper - 1);
  const double dt = p1.time - p0.time;
  if (dt <= 1.0e-9) {
    return p0;
  }
  const double alpha = (time_value - p0.time) / dt;
  GpsPoint out;
  out.time = time_value;
  out.lat = p0.lat + (p1.lat - p0.lat) * alpha;
  out.lon = p0.lon + (p1.lon - p0.lon) * alpha;
  out.bearing = static_cast<float>(p0.bearing + (p1.bearing - p0.bearing) * alpha);
  out.type = alpha < 0.5 ? p0.type : p1.type;
  return out;
}

ImU32 map_timeline_color(TimelineEntry::Type type, float alpha = 1.0f) {
  return timeline_entry_color(type, alpha, {140, 150, 165});
}

ImVec2 gps_to_screen(double lat, double lon, double zoom, double top_left_x, double top_left_y, const ImVec2 &rect_min) {
  return ImVec2(rect_min.x + static_cast<float>(lon_to_world_x(lon, zoom) - top_left_x),
                rect_min.y + static_cast<float>(lat_to_world_y(lat, zoom) - top_left_y));
}

bool point_in_rect_with_margin(const ImVec2 &point, const ImVec2 &rect_min, const ImVec2 &rect_max,
                               float margin_fraction) {
  const float width = rect_max.x - rect_min.x;
  const float height = rect_max.y - rect_min.y;
  const float margin_x = width * margin_fraction;
  const float margin_y = height * margin_fraction;
  return point.x >= rect_min.x + margin_x && point.x <= rect_max.x - margin_x
      && point.y >= rect_min.y + margin_y && point.y <= rect_max.y - margin_y;
}

void draw_car_marker(ImDrawList *draw_list, ImVec2 center, float bearing_deg, ImU32 color, float size) {
  const float rad = bearing_deg * static_cast<float>(M_PI / 180.0);
  const ImVec2 forward(std::sin(rad), -std::cos(rad));
  const ImVec2 perp(-forward.y, forward.x);
  const ImVec2 tip(center.x + forward.x * size, center.y + forward.y * size);
  const ImVec2 base(center.x - forward.x * size * 0.45f, center.y - forward.y * size * 0.45f);
  const ImVec2 left(base.x + perp.x * size * 0.6f, base.y + perp.y * size * 0.6f);
  const ImVec2 right(base.x - perp.x * size * 0.6f, base.y - perp.y * size * 0.6f);
  draw_list->AddTriangleFilled(tip, left, right, color);
  draw_list->AddTriangle(tip, left, right, IM_COL32(255, 255, 255, 210), 2.0f);
}

bool is_convex_ring(const std::vector<ImVec2> &points) {
  if (points.size() < 4) {
    return false;
  }
  float sign = 0.0f;
  const size_t n = points.size();
  for (size_t i = 0; i < n; ++i) {
    const ImVec2 &a = points[i];
    const ImVec2 &b = points[(i + 1) % n];
    const ImVec2 &c = points[(i + 2) % n];
    const float cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x);
    if (std::abs(cross) < 1.0e-3f) {
      continue;
    }
    if (sign == 0.0f) {
      sign = cross;
    } else if ((cross > 0.0f) != (sign > 0.0f)) {
      return false;
    }
  }
  return sign != 0.0f;
}

uint64_t fnv1a64(std::string_view text) {
  uint64_t value = 1469598103934665603ULL;
  for (unsigned char c : text) {
    value ^= static_cast<uint64_t>(c);
    value *= 1099511628211ULL;
  }
  return value;
}

fs::path basemap_cache_root() {
  const char *home = std::getenv("HOME");
  fs::path root = home != nullptr ? fs::path(home) / ".comma" : fs::temp_directory_path();
  root /= "cabana_vector_map";
  fs::create_directories(root);
  return root;
}

std::string bounds_key(const GeoBounds &bounds) {
  return util::string_format("v2_%.5f_%.5f_%.5f_%.5f",
                             bounds.south, bounds.west, bounds.north, bounds.east);
}

fs::path basemap_cache_path(const std::string &key) {
  const uint64_t hash = fnv1a64(key);
  return basemap_cache_root() / util::string_format("%016llx.bin.zst", static_cast<unsigned long long>(hash));
}

uint64_t cache_directory_size_bytes() {
  uint64_t total = 0;
  const fs::path root = basemap_cache_root();
  if (!fs::exists(root)) {
    return 0;
  }
  for (const fs::directory_entry &entry : fs::directory_iterator(root)) {
    if (entry.is_regular_file()) {
      total += static_cast<uint64_t>(entry.file_size());
    }
  }
  return total;
}

size_t cache_directory_file_count() {
  size_t count = 0;
  const fs::path root = basemap_cache_root();
  if (!fs::exists(root)) {
    return 0;
  }
  for (const fs::directory_entry &entry : fs::directory_iterator(root)) {
    if (entry.is_regular_file()) {
      ++count;
    }
  }
  return count;
}

void clear_cache_directory() {
  const fs::path root = basemap_cache_root();
  if (!fs::exists(root)) {
    return;
  }
  for (const fs::directory_entry &entry : fs::directory_iterator(root)) {
    if (entry.is_regular_file()) {
      std::error_code ec;
      fs::remove(entry.path(), ec);
    }
  }
}


std::string bbox_string(const GeoBounds &bounds) {
  return util::string_format("%.6f,%.6f,%.6f,%.6f",
                             bounds.south, bounds.west, bounds.north, bounds.east);
}

MapRequestSpec build_request_for_trace(const GpsTrace &trace) {
  const std::vector<GeoBounds> boxes = corridor_boxes_for_trace(trace);
  GeoBounds union_bounds;
  std::string query = "[out:json][timeout:25];(";
  for (const GeoBounds &box : boxes) {
    union_bounds = merge_bounds(union_bounds, box);
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
  for (const GeoBounds &box : boxes) {
    key += ":";
    key += bbox_string(box);
  }
  return MapRequestSpec{
    .key = std::move(key),
    .bounds = union_bounds,
    .query = std::move(query),
  };
}

bool fetch_overpass_json(std::string_view query, std::string *out, std::atomic<bool> *abort) {
  for (const char *endpoint : MAP_QUERY_ENDPOINTS) {
    if (abort->load()) return false;
    auto response = PyProcess::runModule("openpilot.tools.cabana.core.map_fetch",
                                        {endpoint, std::string(query)}, abort, false);
    if (abort->load()) return false;
    if (!response.empty() && response.front() == '{') { *out = std::move(response); return true; }
  }
  return false;
}
std::string load_overpass_json(std::string_view query, std::atomic<bool> *abort) {
  std::string response;
  fetch_overpass_json(query, &response, abort);
  return response;
}

template <typename T>
void append_pod(std::string *out, const T &value) {
  const size_t start = out->size();
  out->resize(start + sizeof(T));
  std::memcpy(out->data() + start, &value, sizeof(T));
}

template <typename T>
bool read_pod(std::string_view data, size_t *offset, T *value) {
  if (*offset + sizeof(T) > data.size()) {
    return false;
  }
  std::memcpy(value, data.data() + *offset, sizeof(T));
  *offset += sizeof(T);
  return true;
}

void append_points(std::string *out, const std::vector<ProjectedPoint> &points) {
  const uint32_t count = static_cast<uint32_t>(points.size());
  append_pod(out, count);
  for (const ProjectedPoint &point : points) {
    append_pod(out, point.x);
    append_pod(out, point.y);
  }
}

bool read_points(std::string_view data, size_t *offset, std::vector<ProjectedPoint> *points) {
  uint32_t count = 0;
  if (!read_pod(data, offset, &count)) {
    return false;
  }
  points->clear();
  points->reserve(count);
  for (uint32_t i = 0; i < count; ++i) {
    ProjectedPoint point;
    if (!read_pod(data, offset, &point.x) || !read_pod(data, offset, &point.y)) {
      return false;
    }
    points->push_back(point);
  }
  return true;
}

std::string serialize_basemap_payload(const RouteBasemap &basemap) {
  std::string raw;
  raw.reserve(1024 + basemap.roads.size() * 48);
  raw.append("JBM2", 4);
  append_pod(&raw, basemap.bounds.south);
  append_pod(&raw, basemap.bounds.west);
  append_pod(&raw, basemap.bounds.north);
  append_pod(&raw, basemap.bounds.east);

  const uint32_t road_count = static_cast<uint32_t>(basemap.roads.size());
  const uint32_t water_line_count = static_cast<uint32_t>(basemap.water_lines.size());
  const uint32_t water_polygon_count = static_cast<uint32_t>(basemap.water_polygons.size());
  append_pod(&raw, road_count);
  append_pod(&raw, water_line_count);
  append_pod(&raw, water_polygon_count);

  for (const RoadFeature &road : basemap.roads) {
    const uint8_t kind = static_cast<uint8_t>(road.road_class);
    append_pod(&raw, kind);
    append_points(&raw, road.points);
  }
  for (const WaterLineFeature &water : basemap.water_lines) {
    append_points(&raw, water.points);
  }
  for (const WaterPolygonFeature &water : basemap.water_polygons) {
    append_points(&raw, water.ring);
  }
  return raw;
}

std::optional<RouteBasemap> deserialize_basemap_payload(std::string_view raw, const std::string &key) {
  if (!util::starts_with(std::string(raw), "JBM2")) {
    return std::nullopt;
  }
  size_t offset = 4;
  RouteBasemap basemap;
  basemap.key = key;
  if (!read_pod(raw, &offset, &basemap.bounds.south)
      || !read_pod(raw, &offset, &basemap.bounds.west)
      || !read_pod(raw, &offset, &basemap.bounds.north)
      || !read_pod(raw, &offset, &basemap.bounds.east)) {
    return std::nullopt;
  }
  basemap.projected_bounds = project_bounds0(basemap.bounds);

  uint32_t road_count = 0;
  uint32_t water_line_count = 0;
  uint32_t water_polygon_count = 0;
  if (!read_pod(raw, &offset, &road_count)
      || !read_pod(raw, &offset, &water_line_count)
      || !read_pod(raw, &offset, &water_polygon_count)) {
    return std::nullopt;
  }

  basemap.roads.reserve(road_count);
  for (uint32_t i = 0; i < road_count; ++i) {
    uint8_t kind = 0;
    std::vector<ProjectedPoint> points;
    if (!read_pod(raw, &offset, &kind) || !read_points(raw, &offset, &points)) {
      return std::nullopt;
    }
    basemap.roads.push_back(RoadFeature{
      .road_class = static_cast<RoadClass>(kind),
      .bounds = compute_projected_bounds(points),
      .points = std::move(points),
    });
  }

  basemap.water_lines.reserve(water_line_count);
  for (uint32_t i = 0; i < water_line_count; ++i) {
    std::vector<ProjectedPoint> points;
    if (!read_points(raw, &offset, &points)) {
      return std::nullopt;
    }
    basemap.water_lines.push_back(WaterLineFeature{
      .bounds = compute_projected_bounds(points),
      .points = std::move(points),
    });
  }

  basemap.water_polygons.reserve(water_polygon_count);
  for (uint32_t i = 0; i < water_polygon_count; ++i) {
    std::vector<ProjectedPoint> ring;
    if (!read_points(raw, &offset, &ring)) {
      return std::nullopt;
    }
    basemap.water_polygons.push_back(WaterPolygonFeature{
      .bounds = compute_projected_bounds(ring),
      .ring = std::move(ring),
    });
  }
  return basemap;
}

bool save_compressed_basemap(const fs::path &path, const RouteBasemap &basemap) {
  const std::string raw = serialize_basemap_payload(basemap);
  const size_t bound = ZSTD_compressBound(raw.size());
  std::string compressed(bound, '\0');
  const size_t size = ZSTD_compress(compressed.data(), compressed.size(), raw.data(), raw.size(), 5);
  if (ZSTD_isError(size)) {
    return false;
  }
  compressed.resize(size);
  ensure_parent_dir(path);
  const std::string path_string = path.string();
  return util::write_file(path_string.c_str(), compressed.data(), compressed.size(), O_WRONLY | O_CREAT | O_TRUNC) == 0;
}

std::optional<RouteBasemap> load_compressed_basemap(const fs::path &path, const std::string &key) {
  const std::string compressed = util::read_file(path.string());
  if (compressed.empty()) {
    return std::nullopt;
  }
  const unsigned long long raw_size = ZSTD_getFrameContentSize(compressed.data(), compressed.size());
  if (raw_size == ZSTD_CONTENTSIZE_ERROR || raw_size == ZSTD_CONTENTSIZE_UNKNOWN || raw_size > (1ULL << 31)) {
    return std::nullopt;
  }
  std::string raw(static_cast<size_t>(raw_size), '\0');
  const size_t actual = ZSTD_decompress(raw.data(), raw.size(), compressed.data(), compressed.size());
  if (ZSTD_isError(actual)) {
    return std::nullopt;
  }
  raw.resize(actual);
  return deserialize_basemap_payload(raw, key);
}

std::vector<ProjectedPoint> geometry_points(const json11::Json &geometry_json) {
  std::vector<ProjectedPoint> points;
  const auto items = geometry_json.array_items();
  points.reserve(items.size());
  for (const json11::Json &point : items) {
    if (!point["lat"].is_number() || !point["lon"].is_number()) {
      continue;
    }
    points.push_back(ProjectedPoint{
      .x = project_lon0(point["lon"].number_value()),
      .y = project_lat0(point["lat"].number_value()),
    });
  }
  return points;
}

std::optional<RoadClass> classify_road(std::string_view highway) {
  if (highway == "motorway" || highway == "motorway_link" || highway == "trunk" || highway == "trunk_link") {
    return RoadClass::Motorway;
  }
  if (highway == "primary" || highway == "primary_link") {
    return RoadClass::Primary;
  }
  if (highway == "secondary" || highway == "secondary_link" || highway == "tertiary" || highway == "tertiary_link") {
    return RoadClass::Secondary;
  }
  if (highway == "residential" || highway == "unclassified" || highway == "living_street" || highway == "road") {
    return RoadClass::Local;
  }
  return std::nullopt;
}

std::optional<RouteBasemap> parse_basemap_json(const std::string &raw, const GeoBounds &bounds, const std::string &key) {
  std::string parse_error;
  const json11::Json root = json11::Json::parse(raw, parse_error);
  if (!parse_error.empty() || !root.is_object()) {
    return std::nullopt;
  }

  RouteBasemap basemap;
  basemap.key = key;
  basemap.bounds = bounds;
  basemap.projected_bounds = project_bounds0(bounds);

  for (const json11::Json &element : root["elements"].array_items()) {
    if (element["type"].string_value() != "way") {
      continue;
    }
    const json11::Json &tags = element["tags"];
    const std::vector<ProjectedPoint> points = geometry_points(element["geometry"]);
    if (points.size() < 2) {
      continue;
    }

    const std::string highway = tags["highway"].string_value();
    if (!highway.empty()) {
      const std::optional<RoadClass> road_class = classify_road(highway);
      if (!road_class.has_value()) {
        continue;
      }
      basemap.roads.push_back(RoadFeature{
        .road_class = *road_class,
        .bounds = compute_projected_bounds(points),
        .points = points,
      });
      continue;
    }

    const std::string natural = tags["natural"].string_value();
    const std::string waterway = tags["waterway"].string_value();
    const bool closed = points.size() >= 4
                     && std::abs(points.front().x - points.back().x) < 1.0e-6f
                     && std::abs(points.front().y - points.back().y) < 1.0e-6f;
    if ((natural == "water" || waterway == "riverbank") && closed) {
      basemap.water_polygons.push_back(WaterPolygonFeature{
        .bounds = compute_projected_bounds(points),
        .ring = points,
      });
      continue;
    }
    if (waterway == "river" || waterway == "stream" || waterway == "canal") {
      basemap.water_lines.push_back(WaterLineFeature{
        .bounds = compute_projected_bounds(points),
        .points = points,
      });
    }
  }

  return basemap;
}

struct RoadPaint {
  ImU32 casing = 0;
  ImU32 fill = 0;
  float casing_width = 1.0f;
  float fill_width = 1.0f;
};

constexpr ImU32 MAP_BG_COLOR = IM_COL32(244, 243, 238, 255);
constexpr ImU32 MAP_WATER_FILL = IM_COL32(193, 216, 235, 185);
constexpr ImU32 MAP_WATER_OUTLINE = IM_COL32(143, 173, 201, 220);
constexpr ImU32 MAP_WATER_LINE = IM_COL32(156, 186, 214, 205);
constexpr ImU32 MAP_ROUTE_HALO = IM_COL32(31, 40, 50, 92);

RoadPaint road_paint(RoadClass road_class, float zoom) {
  const float scale = std::clamp(0.88f + 0.12f * (zoom - 12.0f), 0.76f, 1.95f);
  switch (road_class) {
    case RoadClass::Motorway:
      return {
        .casing = IM_COL32(163, 157, 149, 235),
        .fill = IM_COL32(245, 235, 215, 255),
        .casing_width = 5.6f * scale,
        .fill_width = 3.7f * scale,
      };
    case RoadClass::Primary:
      return {
        .casing = IM_COL32(171, 171, 168, 220),
        .fill = IM_COL32(249, 246, 237, 248),
        .casing_width = 4.6f * scale,
        .fill_width = 2.95f * scale,
      };
    case RoadClass::Secondary:
      return {
        .casing = IM_COL32(183, 186, 189, 210),
        .fill = IM_COL32(252, 251, 247, 240),
        .casing_width = 3.5f * scale,
        .fill_width = 2.15f * scale,
      };
    case RoadClass::Local:
    default:
      return {
        .casing = IM_COL32(200, 202, 205, 195),
        .fill = IM_COL32(255, 255, 254, 230),
        .casing_width = 2.5f * scale,
        .fill_width = 1.5f * scale,
      };
  }
}

void clamp_map_center(TabUiState::MapPaneState *map_state, const GeoBounds &bounds, const ImVec2 &size) {
  if (!bounds.valid() || size.x <= 1.0f || size.y <= 1.0f) {
    return;
  }
  const double zoom = map_state->zoom;
  const double min_x = lon_to_world_x(bounds.west, zoom);
  const double max_x = lon_to_world_x(bounds.east, zoom);
  const double min_y = lat_to_world_y(bounds.north, zoom);
  const double max_y = lat_to_world_y(bounds.south, zoom);
  const double half_w = size.x * 0.5;
  const double half_h = size.y * 0.5;
  double center_x = lon_to_world_x(map_state->center_lon, zoom);
  double center_y = lat_to_world_y(map_state->center_lat, zoom);
  if (max_x - min_x <= size.x) {
    center_x = (min_x + max_x) * 0.5;
  } else {
    center_x = std::clamp(center_x, min_x + half_w, max_x - half_w);
  }
  if (max_y - min_y <= size.y) {
    center_y = (min_y + max_y) * 0.5;
  } else {
    center_y = std::clamp(center_y, min_y + half_h, max_y - half_h);
  }
  map_state->center_lon = world_x_to_lon(center_x, zoom);
  map_state->center_lat = world_y_to_lat(center_y, zoom);
}

void initialize_map_pane_state(TabUiState::MapPaneState *map_state,
                               const GpsTrace &trace,
                               const GeoBounds &bounds,
                               ImVec2 size,
                               SessionDataMode mode,
                               std::optional<GpsPoint> cursor_point) {
  if (trace.points.empty()) {
    return;
  }
  map_state->initialized = true;
  map_state->follow = mode == SessionDataMode::Stream;
  const int min_zoom = minimum_allowed_map_zoom(bounds, trace, size);
  if (mode == SessionDataMode::Stream && cursor_point.has_value()) {
    map_state->zoom = std::max(16.0f, static_cast<float>(min_zoom));
    map_state->center_lat = cursor_point->lat;
    map_state->center_lon = cursor_point->lon;
  } else {
    map_state->zoom = std::max(static_cast<float>(fit_map_zoom_for_trace(trace, size.x, size.y)),
                               static_cast<float>(min_zoom));
    map_state->center_lat = map_trace_center_lat(trace);
    map_state->center_lon = map_trace_center_lon(trace);
  }
  clamp_map_center(map_state, bounds, size);
}

void draw_feature_polyline(ImDrawList *draw_list,
                           const std::vector<ProjectedPoint> &points,
                           float zoom_scale,
                           double top_left_x,
                           double top_left_y,
                           const ImVec2 &rect_min,
                           ImU32 color,
                           float thickness,
                           bool closed = false) {
  if (points.size() < 2) {
    return;
  }
  std::vector<ImVec2> screen;
  screen.reserve(points.size());
  for (const ProjectedPoint &point : points) {
    screen.push_back(ImVec2(rect_min.x + point.x * zoom_scale - static_cast<float>(top_left_x),
                            rect_min.y + point.y * zoom_scale - static_cast<float>(top_left_y)));
  }
  draw_list->AddPolyline(screen.data(), static_cast<int>(screen.size()), color,
                         closed ? ImDrawFlags_Closed : ImDrawFlags_None, thickness);
}

void draw_water_polygon(ImDrawList *draw_list,
                        const WaterPolygonFeature &feature,
                        float zoom_scale,
                        double top_left_x,
                        double top_left_y,
                        const ImVec2 &rect_min) {
  if (feature.ring.size() < 3) {
    return;
  }
  std::vector<ImVec2> screen;
  screen.reserve(feature.ring.size());
  for (const ProjectedPoint &point : feature.ring) {
    screen.push_back(ImVec2(rect_min.x + point.x * zoom_scale - static_cast<float>(top_left_x),
                            rect_min.y + point.y * zoom_scale - static_cast<float>(top_left_y)));
  }
  if (screen.size() >= 3 && is_convex_ring(screen)) {
    draw_list->AddConvexPolyFilled(screen.data(), static_cast<int>(screen.size()), MAP_WATER_FILL);
  }
  draw_list->AddPolyline(screen.data(), static_cast<int>(screen.size()), MAP_WATER_OUTLINE,
                         ImDrawFlags_Closed, 1.8f);
}

void draw_edge_fade(ImDrawList *draw_list,
                    const GeoBounds &bounds,
                    double zoom,
                    double top_left_x,
                    double top_left_y,
                    const ImVec2 &rect_min,
                    const ImVec2 &rect_max) {
  if (!bounds.valid()) {
    return;
  }

  const float west_x = rect_min.x + static_cast<float>(lon_to_world_x(bounds.west, zoom) - top_left_x);
  const float east_x = rect_min.x + static_cast<float>(lon_to_world_x(bounds.east, zoom) - top_left_x);
  const float north_y = rect_min.y + static_cast<float>(lat_to_world_y(bounds.north, zoom) - top_left_y);
  const float south_y = rect_min.y + static_cast<float>(lat_to_world_y(bounds.south, zoom) - top_left_y);

  const float fade_x = std::max(28.0f, (rect_max.x - rect_min.x) * MAP_EDGE_FADE_FRAC);
  const float fade_y = std::max(28.0f, (rect_max.y - rect_min.y) * MAP_EDGE_FADE_FRAC);
  const ImU32 solid = MAP_BG_COLOR;
  const ImU32 clear = IM_COL32(244, 243, 238, 6);

  if (west_x > rect_min.x) {
    const float x0 = rect_min.x;
    const float x1 = std::min(rect_max.x, west_x);
    const float xfade = std::max(x0, x1 - fade_x);
    draw_list->AddRectFilledMultiColor(ImVec2(x0, rect_min.y), ImVec2(xfade, rect_max.y), solid, solid, solid, solid);
    draw_list->AddRectFilledMultiColor(ImVec2(xfade, rect_min.y), ImVec2(x1, rect_max.y), solid, clear, clear, solid);
  }
  if (east_x < rect_max.x) {
    const float x0 = std::max(rect_min.x, east_x);
    const float x1 = rect_max.x;
    const float xfade = std::min(x1, x0 + fade_x);
    draw_list->AddRectFilledMultiColor(ImVec2(x0, rect_min.y), ImVec2(xfade, rect_max.y), clear, solid, solid, clear);
    draw_list->AddRectFilledMultiColor(ImVec2(xfade, rect_min.y), ImVec2(x1, rect_max.y), solid, solid, solid, solid);
  }
  if (north_y > rect_min.y) {
    const float y0 = rect_min.y;
    const float y1 = std::min(rect_max.y, north_y);
    const float yfade = std::max(y0, y1 - fade_y);
    draw_list->AddRectFilledMultiColor(ImVec2(rect_min.x, y0), ImVec2(rect_max.x, yfade), solid, solid, solid, solid);
    draw_list->AddRectFilledMultiColor(ImVec2(rect_min.x, yfade), ImVec2(rect_max.x, y1), solid, solid, clear, clear);
  }
  if (south_y < rect_max.y) {
    const float y0 = std::max(rect_min.y, south_y);
    const float y1 = rect_max.y;
    const float yfade = std::min(y1, y0 + fade_y);
    draw_list->AddRectFilledMultiColor(ImVec2(rect_min.x, y0), ImVec2(rect_max.x, yfade), clear, clear, solid, solid);
    draw_list->AddRectFilledMultiColor(ImVec2(rect_min.x, yfade), ImVec2(rect_max.x, y1), solid, solid, solid, solid);
  }
}

}  // namespace

MapDataManager::MapDataManager() : worker_([this]() { run(); }) {}

MapDataManager::~MapDataManager() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = true;
  }
  cv_.notify_all();
  if (worker_.joinable()) {
    worker_.join();
  }
}

void MapDataManager::pump() {
  std::unique_ptr<RouteBasemap> ready;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    ready = std::move(completed_);
  }
  if (ready) {
    current_ = std::move(ready);
  }
}

void MapDataManager::ensureTrace(const GpsTrace &trace) {
  if (trace.points.empty()) {
    return;
  }
  const MapRequestSpec wanted = build_request_for_trace(trace);
  if (!wanted.bounds.valid()) {
    return;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  if ((current_ && current_->key == wanted.key) || (pending_ && pending_->key == wanted.key) || (active_ && active_->key == wanted.key)) {
    return;
  }

  if (failed_key_ == wanted.key && std::chrono::steady_clock::now() < retry_after_) return;
  error_.clear();

  pending_ = std::make_unique<Request>(Request{
    .key = wanted.key,
    .bounds = wanted.bounds,
    .query = wanted.query,
  });
  cv_.notify_one();
}

bool MapDataManager::loading() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return active_ || pending_;
}

const RouteBasemap *MapDataManager::current() const {
  return current_.get();
}

std::string MapDataManager::error() const { std::lock_guard<std::mutex> lock(mutex_); return error_; }

void MapDataManager::clearCache() {
  std::lock_guard<std::mutex> lock(mutex_);
  try { clear_cache_directory(); failed_key_.clear(); }
  catch (const std::exception &failure) { error_ = failure.what(); }
}

MapCacheStats MapDataManager::cacheStats() const {
  try { return MapCacheStats{.bytes = cache_directory_size_bytes(), .files = cache_directory_file_count()}; }
  catch (const std::exception &) { return {}; }
}

void MapDataManager::run() {
  while (true) {
    Request request;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [&]() { return stopping_.load() || pending_ != nullptr; });
      if (stopping_) {
        return;
      }
      request = *pending_;
      active_ = std::move(pending_);
    }

    std::unique_ptr<RouteBasemap> parsed;
    std::string failure;
    try {
      if (auto cached = load_compressed_basemap(basemap_cache_path(request.key), request.key)) {
        parsed = std::make_unique<RouteBasemap>(std::move(*cached));
      } else {
        const std::string raw = load_overpass_json(request.query, &stopping_);
        if (!raw.empty()) {
          if (auto basemap = parse_basemap_json(raw, request.bounds, request.key)) {
            parsed = std::make_unique<RouteBasemap>(std::move(*basemap));
            try { save_compressed_basemap(basemap_cache_path(request.key), *parsed); }
            catch (const std::exception &error) { failure = std::string("Map cache: ") + error.what(); }
          }
        }
        if (!parsed) failure = "Road data unavailable; GPS trace remains usable. Retrying in 30 seconds.";
      }
    } catch (const std::exception &error) { failure = error.what(); }

    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (active_ && active_->key == request.key) {
        error_ = failure;
        if (!parsed) { failed_key_ = request.key; retry_after_ = std::chrono::steady_clock::now() + std::chrono::seconds(30); }
        else failed_key_.clear();
        completed_ = std::move(parsed);
        active_.reset();
      }
    }
  }
}

void draw_map_pane(AppSession *session, UiState *state, void *, int pane_index) {
  TabUiState *tab_state = app_active_tab_state(state);
  if (tab_state == nullptr || pane_index < 0 || pane_index >= static_cast<int>(tab_state->map_panes.size())) {
    ImGui::TextUnformatted("Map unavailable");
    return;
  }
  if (!session->map_data) {
    ImGui::TextUnformatted("Map unavailable");
    return;
  }

  session->map_data->pump();

  TabUiState::MapPaneState &map_state = tab_state->map_panes[static_cast<size_t>(pane_index)];
  const GpsTrace &trace = session->route_data.gps_trace;
  const RouteBasemap *basemap = session->map_data->current();
  const GeoBounds map_bounds = basemap != nullptr ? basemap->bounds : requested_bounds_for_trace(trace);

  const ImVec2 rect_min = ImGui::GetCursorScreenPos();
  const ImVec2 size = ImGui::GetContentRegionAvail();
  const ImVec2 input_size(std::max(1.0f, size.x - 22.0f), std::max(1.0f, size.y));
  ImGui::SetNextItemAllowOverlap();
  ImGui::InvisibleButton("##map_canvas", input_size);
  const ImVec2 rect_max(rect_min.x + size.x, rect_min.y + size.y);
  const float rect_width = rect_max.x - rect_min.x;
  const float rect_height = rect_max.y - rect_min.y;
  ImDrawList *draw_list = ImGui::GetWindowDrawList();

  draw_list->PushClipRect(rect_min, rect_max, true);
  draw_list->AddRectFilled(rect_min, rect_max, MAP_BG_COLOR);

  if (trace.points.empty()) {
    const char *label = session->async_route_loading ? "Loading map..." : "No GPS trace";
    const ImVec2 text = ImGui::CalcTextSize(label);
    draw_list->AddText(ImVec2(rect_min.x + (rect_width - text.x) * 0.5f,
                              rect_min.y + (rect_height - text.y) * 0.5f),
                       IM_COL32(110, 118, 128, 255), label);
    draw_list->PopClipRect();
    return;
  }

  const std::optional<GpsPoint> cursor_point = state->has_tracker_time
    ? interpolate_gps(trace, state->tracker_time)
    : std::optional<GpsPoint>{};
  if (!map_state.initialized) {
    initialize_map_pane_state(&map_state, trace, map_bounds, size, session->data_mode, cursor_point);
  }

  const int min_zoom = minimum_allowed_map_zoom(map_bounds, trace, size);
  if (map_state.follow && cursor_point.has_value()) {
    const float follow_zoom = std::clamp(map_state.zoom, static_cast<float>(min_zoom), static_cast<float>(MAP_MAX_ZOOM));
    const double center_x = lon_to_world_x(map_state.center_lon, follow_zoom);
    const double center_y = lat_to_world_y(map_state.center_lat, follow_zoom);
    const double top_left_x = center_x - rect_width * 0.5;
    const double top_left_y = center_y - rect_height * 0.5;
    const ImVec2 car_screen = gps_to_screen(cursor_point->lat, cursor_point->lon, follow_zoom, top_left_x, top_left_y, rect_min);
    if (!point_in_rect_with_margin(car_screen, rect_min, rect_max, 0.22f)) {
      map_state.center_lat = cursor_point->lat;
      map_state.center_lon = cursor_point->lon;
    }
  }

  map_state.zoom = std::clamp(map_state.zoom, static_cast<float>(min_zoom), static_cast<float>(MAP_MAX_ZOOM));
  clamp_map_center(&map_state, map_bounds, size);

  const double zoom = map_state.zoom;
  const float zoom_scale = static_cast<float>(std::exp2(zoom));
  const double center_x = lon_to_world_x(map_state.center_lon, zoom);
  const double center_y = lat_to_world_y(map_state.center_lat, zoom);
  const double top_left_x = center_x - rect_width * 0.5;
  const double top_left_y = center_y - rect_height * 0.5;
  const ProjectedBounds current_view = view_bounds(top_left_x, top_left_y, rect_width, rect_height);

  if (basemap != nullptr) {
    for (const WaterPolygonFeature &water : basemap->water_polygons) {
      if (feature_intersects_view(water.bounds, current_view, zoom_scale)) {
        draw_water_polygon(draw_list, water, zoom_scale, top_left_x, top_left_y, rect_min);
      }
    }
    for (const WaterLineFeature &water : basemap->water_lines) {
      if (feature_intersects_view(water.bounds, current_view, zoom_scale)) {
        draw_feature_polyline(draw_list, water.points, zoom_scale, top_left_x, top_left_y, rect_min,
                              MAP_WATER_LINE, 2.4f);
      }
    }

    std::array<RoadClass, 4> order = {
      RoadClass::Local,
      RoadClass::Secondary,
      RoadClass::Primary,
      RoadClass::Motorway,
    };
    for (RoadClass road_class : order) {
      const RoadPaint paint = road_paint(road_class, static_cast<float>(zoom));
      for (const RoadFeature &road : basemap->roads) {
        if (road.road_class != road_class || !feature_intersects_view(road.bounds, current_view, zoom_scale)) {
          continue;
        }
        draw_feature_polyline(draw_list, road.points, zoom_scale, top_left_x, top_left_y, rect_min,
                              paint.casing, paint.casing_width);
        draw_feature_polyline(draw_list, road.points, zoom_scale, top_left_x, top_left_y, rect_min,
                              paint.fill, paint.fill_width);
      }
    }
  }

  if (basemap != nullptr) {
    draw_edge_fade(draw_list, basemap->bounds, zoom, top_left_x, top_left_y, rect_min, rect_max);
  }

  for (size_t i = 1; i < trace.points.size(); ++i) {
    const GpsPoint &p0 = trace.points[i - 1];
    const GpsPoint &p1 = trace.points[i];
    const ImVec2 s0 = gps_to_screen(p0.lat, p0.lon, zoom, top_left_x, top_left_y, rect_min);
    const ImVec2 s1 = gps_to_screen(p1.lat, p1.lon, zoom, top_left_x, top_left_y, rect_min);
    draw_list->AddLine(s0, s1, MAP_ROUTE_HALO, 5.8f);
    draw_list->AddLine(s0, s1, map_timeline_color(p1.type, 1.0f), 3.25f);
  }

  if (cursor_point.has_value()) {
    const ImVec2 marker = gps_to_screen(cursor_point->lat, cursor_point->lon, zoom, top_left_x, top_left_y, rect_min);
    const float marker_size = std::clamp(9.0f + 1.0f * static_cast<float>(zoom - min_zoom), 9.0f, 20.0f);
    draw_car_marker(draw_list, marker, cursor_point->bearing, map_timeline_color(cursor_point->type, 1.0f), marker_size);
  }

  const std::string map_error = session->map_data->error();
  if (session->map_data->loading() || !map_error.empty()) {
    const char *label = !map_error.empty() ? map_error.c_str() : basemap != nullptr ? "Refreshing roads..." : "Loading roads...";
    const ImVec2 text = ImGui::CalcTextSize(label);
    const ImVec2 pos(rect_min.x + 12.0f, rect_max.y - text.y - 12.0f);
    draw_list->AddRectFilled(ImVec2(pos.x - 6.0f, pos.y - 4.0f),
                             ImVec2(pos.x + text.x + 6.0f, pos.y + text.y + 4.0f),
                             IM_COL32(255, 255, 255, 180), 4.0f);
    draw_list->AddText(pos, IM_COL32(84, 93, 105, 255), label);
  }
  draw_list->PopClipRect();

  const bool canvas_hovered = ImGui::IsItemHovered();
  const bool double_clicked = canvas_hovered && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left);
  bool overlay_hovered = false;
  if (const std::string google_maps_url = route_google_maps_url(trace); !google_maps_url.empty()) {
    std::string label = std::string("Google Maps ") + icon::BOX_ARROW_UP_RIGHT;
    const ImVec2 text_size = ImGui::CalcTextSize(label.c_str());
    const ImVec2 button_size(text_size.x + 20.0f, text_size.y + 10.0f);
    const ImVec2 button_pos(rect_max.x - button_size.x - 28.0f, rect_min.y + 10.0f);
    ImGui::SetCursorScreenPos(button_pos);
    ImGui::SetNextItemAllowOverlap();
    if (ImGui::Button("##open_google_maps", button_size)) {
      open_external_url(google_maps_url);
      state->status_text = "Opened Google Maps";
    }
    overlay_hovered = ImGui::IsItemHovered();
    draw_list->AddText(ImVec2(button_pos.x + 10.0f, button_pos.y + (button_size.y - text_size.y) * 0.5f),
                       ImGui::GetColorU32(ImGuiCol_Text), label.c_str());
  }
  const bool hovered = canvas_hovered && !overlay_hovered;
  if (hovered && ImGui::GetIO().MouseWheel != 0.0f) {
    const float next_zoom = std::clamp(static_cast<float>(zoom) + ImGui::GetIO().MouseWheel * MAP_WHEEL_ZOOM_STEP,
                                       static_cast<float>(min_zoom), static_cast<float>(MAP_MAX_ZOOM));
    if (std::abs(next_zoom - zoom) > 1.0e-4f) {
      const ImVec2 mouse = ImGui::GetIO().MousePos;
      const double mouse_world_x = top_left_x + (mouse.x - rect_min.x);
      const double mouse_world_y = top_left_y + (mouse.y - rect_min.y);
      const double mouse_lon = world_x_to_lon(mouse_world_x, zoom);
      const double mouse_lat = world_y_to_lat(mouse_world_y, zoom);
      const double next_center_x = lon_to_world_x(mouse_lon, next_zoom) - (mouse.x - rect_min.x) + rect_width * 0.5;
      const double next_center_y = lat_to_world_y(mouse_lat, next_zoom) - (mouse.y - rect_min.y) + rect_height * 0.5;
      map_state.zoom = next_zoom;
      map_state.center_lon = world_x_to_lon(next_center_x, next_zoom);
      map_state.center_lat = world_y_to_lat(next_center_y, next_zoom);
      map_state.follow = false;
      clamp_map_center(&map_state, map_bounds, size);
    }
  }
  if (hovered && ImGui::IsMouseDragging(ImGuiMouseButton_Left, 2.0f)) {
    const ImVec2 delta = ImGui::GetIO().MouseDelta;
    const double next_center_x = center_x - delta.x;
    const double next_center_y = center_y - delta.y;
    map_state.center_lon = world_x_to_lon(next_center_x, zoom);
    map_state.center_lat = world_y_to_lat(next_center_y, zoom);
    map_state.follow = false;
    clamp_map_center(&map_state, map_bounds, size);
  } else if (hovered && ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
    const ImVec2 drag_delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left);
    if (drag_delta.x * drag_delta.x + drag_delta.y * drag_delta.y < 16.0f) {
      const ImVec2 mouse = ImGui::GetIO().MousePos;
      double best_dist = std::numeric_limits<double>::max();
      double best_time = state->tracker_time;
      for (const GpsPoint &point : trace.points) {
        const ImVec2 screen = gps_to_screen(point.lat, point.lon, zoom, top_left_x, top_left_y, rect_min);
        const double dx = static_cast<double>(screen.x - mouse.x);
        const double dy = static_cast<double>(screen.y - mouse.y);
        const double dist = dx * dx + dy * dy;
        if (dist < best_dist) {
          best_dist = dist;
          best_time = point.time;
        }
      }
      state->tracker_time = best_time;
      state->has_tracker_time = true;
    }
    ImGui::ResetMouseDragDelta(ImGuiMouseButton_Left);
  }
  if (double_clicked) {
    map_state.initialized = false;
  }
}

}  // namespace map_detail
struct MapWidget::Impl {
  map_detail::AppSession session;
  map_detail::UiState state;
  Impl() {
    static std::weak_ptr<map_detail::MapDataManager> shared;
    session.map_data = shared.lock();
    if (!session.map_data) { session.map_data = std::make_shared<map_detail::MapDataManager>(); shared = session.map_data; }
  }
};
MapWidget::MapWidget() : impl_(std::make_unique<Impl>()) {}
MapWidget::~MapWidget() = default;
void MapWidget::setSeries(const std::map<std::string, CerealPoints> &series, uint64_t origin) {
  using namespace map_detail;
  GpsTrace trace;
  const auto sample = [&](const std::string &path, uint64_t time) -> std::optional<double> {
    auto it = series.find(path);
    return it == series.end() ? std::nullopt : heldValue(it->second, time);
  };
  for (const std::string service : {"gpsLocationExternal", "gpsLocation", "liveLocationKalman"}) {
    const bool kalman = service == "liveLocationKalman";
    const std::string latitude = service + (kalman ? "/positionGeodetic/value/0" : "/latitude");
    const std::string longitude = service + (kalman ? "/positionGeodetic/value/1" : "/longitude");
    auto found = series.find(latitude);
    if (found == series.end()) continue;
    for (const auto &point : found->second) {
      auto lon = sample(longitude, point.mono_time);
      auto valid = sample(service + (kalman ? "/positionGeodetic/valid" : "/hasFix"), point.mono_time);
      if (valid && *valid == 0) continue;
      if (!lon || !std::isfinite(*lon) || !std::isfinite(point.value) || point.value < -90 || point.value > 90 || *lon < -180 || *lon > 180 || point.mono_time < origin) continue;
      GpsPoint gps;
      gps.time = (point.mono_time - origin) * 1e-9;
      gps.lat = point.value; gps.lon = *lon;
      auto bearing = sample(service + "/bearingDeg", point.mono_time);
      gps.bearing = bearing && std::isfinite(*bearing) ? *bearing : 0;
      auto enabled = sample("selfdriveState/enabled", point.mono_time);
      if (enabled && *enabled == 1) gps.type = TimelineEntry::Type::Engaged;
      if (trace.points.empty()) { trace.min_lat = trace.max_lat = gps.lat; trace.min_lon = trace.max_lon = gps.lon; }
      trace.min_lat = std::min(trace.min_lat, gps.lat); trace.max_lat = std::max(trace.max_lat, gps.lat);
      trace.min_lon = std::min(trace.min_lon, gps.lon); trace.max_lon = std::max(trace.max_lon, gps.lon);
      trace.points.push_back(gps);
    }
    if (!trace.points.empty()) break;
  }
  impl_->session.route_data.gps_trace = std::move(trace);
  impl_->session.map_data->ensureTrace(impl_->session.route_data.gps_trace);
}
std::optional<double> MapWidget::draw(double seconds, bool live) {
  impl_->state.tracker_time = seconds;
  impl_->session.data_mode = live ? map_detail::SessionDataMode::Stream : map_detail::SessionDataMode::Route;
  map_detail::draw_map_pane(&impl_->session, &impl_->state, nullptr, 0);
  return impl_->state.tracker_time != seconds ? std::optional<double>(impl_->state.tracker_time) : std::nullopt;
}
}  // namespace cabana
