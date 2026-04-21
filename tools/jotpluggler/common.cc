#include "tools/jotpluggler/common.h"

#include <algorithm>
#include <array>
#include <cstdlib>

namespace {

std::string format_coord(const GpsPoint &point) {
  return util::string_format("%.5f,%.5f", point.lat, point.lon);
}

}  // namespace

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

const SpecialItemSpec *special_item_spec(std::string_view item_id) {
  auto it = std::find_if(kSpecialItemSpecs.begin(), kSpecialItemSpecs.end(), [&](const SpecialItemSpec &spec) {
    return item_id == spec.id;
  });
  return it != kSpecialItemSpecs.end() ? &*it : nullptr;
}

const char *special_item_label(std::string_view item_id) {
  const SpecialItemSpec *spec = special_item_spec(item_id);
  return spec != nullptr ? spec->label : "Item";
}

bool pane_kind_is_special(PaneKind kind) {
  return kind == PaneKind::Map || kind == PaneKind::Camera;
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

const std::filesystem::path &repo_root() {
  static const std::filesystem::path root(JOTP_REPO_ROOT);
  return root;
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
  static constexpr const char *kLabels[] = {
    "disengaged",
    "engaged",
    "alert info",
    "alert warning",
    "alert critical",
  };
  const size_t index = static_cast<size_t>(type);
  return index < std::size(kLabels) ? kLabels[index] : kLabels[0];
}

TimelineEntry::Type timeline_type_at_time(const std::vector<TimelineEntry> &timeline, double time_value) {
  for (const TimelineEntry &entry : timeline) {
    if (time_value >= entry.start_time && time_value <= entry.end_time) {
      return entry.type;
    }
  }
  return TimelineEntry::Type::None;
}

std::string normalize_stream_address(std::string address) {
  return is_local_stream_address(address) ? "127.0.0.1" : address;
}

const char *stream_source_kind_label(StreamSourceKind kind) {
  static constexpr const char *kLabels[] = {
    "Local (MSGQ)",
    "Remote (ZMQ)",
  };
  const size_t index = static_cast<size_t>(kind);
  return index < std::size(kLabels) ? kLabels[index] : kLabels[0];
}

std::string stream_source_target_label(const StreamSourceConfig &source) {
  switch (source.kind) {
    case StreamSourceKind::CerealRemote:
      return normalize_stream_address(source.address);
    case StreamSourceKind::CerealLocal:
    default:
      return "127.0.0.1";
  }
}

bool env_flag_enabled(const char *name, bool default_value) {
  const char *raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') {
    return default_value;
  }
  const std::string value = lowercase_copy(util::strip(raw));
  return !(value == "0" || value == "false" || value == "no" || value == "off");
}

void open_external_url(std::string_view url) {
#ifdef __APPLE__
  const std::string command = "open " + shell_quote(url) + " &";
#else
  const std::string command = "xdg-open " + shell_quote(url) + " >/dev/null 2>&1 &";
#endif
  util::check_system(command);
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
