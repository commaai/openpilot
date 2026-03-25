#include "tools/jotpluggler/app_common.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>

namespace {

constexpr std::array<CameraViewSpec, 4> kCameraViewSpecs = {{
  {CameraViewKind::Road,     "Road Camera",      "road",  "road",      "camera_road",      &RouteData::road_camera},
  {CameraViewKind::Driver,   "Driver Camera",    "driver","driver",    "camera_driver",    &RouteData::driver_camera},
  {CameraViewKind::WideRoad, "Wide Road Camera", "wide",  "wide_road", "camera_wide_road", &RouteData::wide_road_camera},
  {CameraViewKind::QRoad,    "qRoad Camera",     "qroad", "qroad",     "camera_qroad",     &RouteData::qroad_camera},
}};

std::string format_coord(const GpsPoint &point) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.5f,%.5f", point.lat, point.lon);
  return std::string(buf);
}

}  // namespace

const std::array<CameraViewSpec, 4> &camera_view_specs() {
  return kCameraViewSpecs;
}

const CameraViewSpec &camera_view_spec(CameraViewKind view) {
  auto it = std::find_if(kCameraViewSpecs.begin(), kCameraViewSpecs.end(), [&](const CameraViewSpec &spec) {
    return spec.view == view;
  });
  return it != kCameraViewSpecs.end() ? *it : kCameraViewSpecs.front();
}

const CameraViewSpec *camera_view_spec_from_special_item(std::string_view item_id) {
  auto it = std::find_if(kCameraViewSpecs.begin(), kCameraViewSpecs.end(), [&](const CameraViewSpec &spec) {
    return item_id == spec.special_item_id;
  });
  return it != kCameraViewSpecs.end() ? &*it : nullptr;
}

const CameraViewSpec *camera_view_spec_from_layout_name(std::string_view layout_name) {
  auto it = std::find_if(kCameraViewSpecs.begin(), kCameraViewSpecs.end(), [&](const CameraViewSpec &spec) {
    return layout_name == spec.layout_name;
  });
  return it != kCameraViewSpecs.end() ? &*it : nullptr;
}

const std::array<SpecialItemSpec, 5> &special_item_specs() {
  static const std::array<SpecialItemSpec, 5> specs = [] {
    std::array<SpecialItemSpec, 5> out = {{
      {"map", "Map", PaneKind::Map, CameraViewKind::Road},
      {},
      {},
      {},
      {},
    }};
    for (size_t i = 0; i < kCameraViewSpecs.size(); ++i) {
      out[i + 1] = SpecialItemSpec{
        kCameraViewSpecs[i].special_item_id,
        kCameraViewSpecs[i].label,
        PaneKind::Camera,
        kCameraViewSpecs[i].view,
      };
    }
    return out;
  }();
  return specs;
}

const SpecialItemSpec *special_item_spec(std::string_view item_id) {
  const auto &specs = special_item_specs();
  auto it = std::find_if(specs.begin(), specs.end(), [&](const SpecialItemSpec &spec) {
    return item_id == spec.id;
  });
  return it != specs.end() ? &*it : nullptr;
}

const char *special_item_label(std::string_view item_id) {
  const SpecialItemSpec *spec = special_item_spec(item_id);
  return spec != nullptr ? spec->label : "Item";
}

bool pane_kind_is_special(PaneKind kind) {
  return kind == PaneKind::Map || kind == PaneKind::Camera;
}

bool pane_is_special(const Pane &pane) {
  return pane_kind_is_special(pane.kind);
}

bool is_default_special_title(std::string_view title) {
  if (title == "Map") return true;
  return std::any_of(kCameraViewSpecs.begin(), kCameraViewSpecs.end(), [&](const CameraViewSpec &spec) {
    return title == spec.label;
  });
}

CameraViewKind sidebar_preview_camera_view(const AppSession &session) {
  return session.route_data.road_camera.entries.empty() && !session.route_data.qroad_camera.entries.empty()
    ? CameraViewKind::QRoad
    : CameraViewKind::Road;
}

ImU32 timeline_entry_color(TimelineEntry::Type type, float alpha) {
  return timeline_entry_color(type, alpha, {111, 143, 175});
}

ImU32 timeline_entry_color(TimelineEntry::Type type, float alpha, std::array<uint8_t, 3> none_color) {
  switch (type) {
    case TimelineEntry::Type::Engaged:
      return ImGui::GetColorU32(color_rgb(0, 163, 108, alpha));
    case TimelineEntry::Type::AlertInfo:
      return ImGui::GetColorU32(color_rgb(255, 195, 0, alpha));
    case TimelineEntry::Type::AlertWarning:
    case TimelineEntry::Type::AlertCritical:
      return ImGui::GetColorU32(color_rgb(199, 0, 57, alpha));
    case TimelineEntry::Type::None:
    default:
      return ImGui::GetColorU32(color_rgb(none_color, alpha));
  }
}

const char *timeline_entry_label(TimelineEntry::Type type) {
  switch (type) {
    case TimelineEntry::Type::Engaged:
      return "engaged";
    case TimelineEntry::Type::AlertInfo:
      return "alert info";
    case TimelineEntry::Type::AlertWarning:
      return "alert warning";
    case TimelineEntry::Type::AlertCritical:
      return "alert critical";
    case TimelineEntry::Type::None:
    default:
      return "disengaged";
  }
}

TimelineEntry::Type timeline_type_at_time(const std::vector<TimelineEntry> &timeline, double time_value) {
  for (const TimelineEntry &entry : timeline) {
    if (time_value >= entry.start_time && time_value <= entry.end_time) {
      return entry.type;
    }
  }
  return TimelineEntry::Type::None;
}

bool env_flag_enabled(const char *name, bool default_value) {
  const char *raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') {
    return default_value;
  }
  const std::string value = lowercase(trim_copy(raw));
  return !(value == "0" || value == "false" || value == "no" || value == "off");
}

void open_external_url(std::string_view url) {
#ifdef __APPLE__
  const std::string command = "open " + shell_quote(url) + " &";
#else
  const std::string command = "xdg-open " + shell_quote(url) + " >/dev/null 2>&1 &";
#endif
  const int ret = std::system(command.c_str());
  (void)ret;
}

std::string route_useradmin_url(const RouteIdentifier &route_id) {
  return route_id.empty() ? std::string()
                          : "https://useradmin.comma.ai/?onebox=" + route_id.dongle_id + "%7C" + route_id.log_id;
}

std::string route_connect_url(const RouteIdentifier &route_id) {
  return route_id.empty() ? std::string()
                          : "https://connect.comma.ai/" + route_id.canonical();
}

std::string route_google_maps_url(const GpsTrace &trace) {
  if (trace.points.size() < 2) {
    return {};
  }

  const std::string prefix = "https://www.google.com/maps/dir/?api=1&travelmode=driving&origin="
                           + format_coord(trace.points.front()) + "&destination=" + format_coord(trace.points.back());
  for (size_t n = std::min<size_t>(9, trace.points.size() > 2 ? trace.points.size() - 2 : 0); ; --n) {
    std::string url = prefix;
    if (n > 0) {
      url += "&waypoints=";
      for (size_t i = 0; i < n; ++i) {
        if (i) url += "%7C";
        url += format_coord(trace.points[1 + ((trace.points.size() - 2) * (i + 1)) / (n + 1)]);
      }
    }
    if (url.size() <= 1900 || n == 0) return url;
  }
}
