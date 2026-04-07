#pragma once

#include "tools/jotpluggler/app.h"

#include <array>
#include <string_view>

struct CameraViewSpec {
  CameraViewKind view = CameraViewKind::Road;
  const char *label = "";
  const char *runtime_name = "";
  const char *layout_name = "";
  const char *special_item_id = "";
  CameraFeedIndex RouteData::*route_member = nullptr;
};

struct SpecialItemSpec {
  const char *id = "";
  const char *label = "";
  PaneKind kind = PaneKind::Plot;
  CameraViewKind camera_view = CameraViewKind::Road;
};

inline constexpr std::array<CameraViewSpec, 4> kCameraViewSpecs = {{
  {CameraViewKind::Road,     "Road Camera",      "road",   "road",      "camera_road",      &RouteData::road_camera},
  {CameraViewKind::Driver,   "Driver Camera",    "driver", "driver",    "camera_driver",    &RouteData::driver_camera},
  {CameraViewKind::WideRoad, "Wide Road Camera", "wide",   "wide_road", "camera_wide_road", &RouteData::wide_road_camera},
  {CameraViewKind::QRoad,    "qRoad Camera",     "qroad",  "qroad",     "camera_qroad",     &RouteData::qroad_camera},
}};

inline constexpr std::array<SpecialItemSpec, 5> kSpecialItemSpecs = {{
  {"map", "Map", PaneKind::Map, CameraViewKind::Road},
  {kCameraViewSpecs[0].special_item_id, kCameraViewSpecs[0].label, PaneKind::Camera, kCameraViewSpecs[0].view},
  {kCameraViewSpecs[1].special_item_id, kCameraViewSpecs[1].label, PaneKind::Camera, kCameraViewSpecs[1].view},
  {kCameraViewSpecs[2].special_item_id, kCameraViewSpecs[2].label, PaneKind::Camera, kCameraViewSpecs[2].view},
  {kCameraViewSpecs[3].special_item_id, kCameraViewSpecs[3].label, PaneKind::Camera, kCameraViewSpecs[3].view},
}};

const CameraViewSpec &camera_view_spec(CameraViewKind view);
const CameraViewSpec *camera_view_spec_from_special_item(std::string_view item_id);
const CameraViewSpec *camera_view_spec_from_layout_name(std::string_view layout_name);

const SpecialItemSpec *special_item_spec(std::string_view item_id);
const char *special_item_label(std::string_view item_id);

bool pane_kind_is_special(PaneKind kind);
bool is_default_special_title(std::string_view title);
CameraViewKind sidebar_preview_camera_view(const AppSession &session);
const std::filesystem::path &repo_root();

ImU32 timeline_entry_color(TimelineEntry::Type type, float alpha = 1.0f);
ImU32 timeline_entry_color(TimelineEntry::Type type, float alpha, std::array<uint8_t, 3> none_color);
const char *timeline_entry_label(TimelineEntry::Type type);
TimelineEntry::Type timeline_type_at_time(const std::vector<TimelineEntry> &timeline, double time_value);
std::string normalize_stream_address(std::string address);
const char *stream_source_kind_label(StreamSourceKind kind);
std::string stream_source_target_label(const StreamSourceConfig &source);

bool env_flag_enabled(const char *name, bool default_value = false);
void open_external_url(std::string_view url);
std::string route_useradmin_url(const RouteIdentifier &route_id);
std::string route_connect_url(const RouteIdentifier &route_id);
std::string route_google_maps_url(const GpsTrace &trace);
