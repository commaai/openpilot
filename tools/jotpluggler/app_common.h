#pragma once

#include "tools/jotpluggler/jotpluggler.h"

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

const std::array<CameraViewSpec, 4> &camera_view_specs();
const CameraViewSpec &camera_view_spec(CameraViewKind view);
const CameraViewSpec *camera_view_spec_from_special_item(std::string_view item_id);
const CameraViewSpec *camera_view_spec_from_layout_name(std::string_view layout_name);

const std::array<SpecialItemSpec, 5> &special_item_specs();
const SpecialItemSpec *special_item_spec(std::string_view item_id);
const char *special_item_label(std::string_view item_id);

bool pane_is_special(const Pane &pane);
bool pane_kind_is_special(PaneKind kind);
bool is_default_special_title(std::string_view title);
CameraViewKind sidebar_preview_camera_view(const AppSession &session);

ImU32 timeline_entry_color(TimelineEntry::Type type, float alpha = 1.0f);
ImU32 timeline_entry_color(TimelineEntry::Type type, float alpha, std::array<uint8_t, 3> none_color);
const char *timeline_entry_label(TimelineEntry::Type type);
TimelineEntry::Type timeline_type_at_time(const std::vector<TimelineEntry> &timeline, double time_value);

bool env_flag_enabled(const char *name, bool default_value = false);
void open_external_url(std::string_view url);
std::string route_useradmin_url(const RouteIdentifier &route_id);
std::string route_connect_url(const RouteIdentifier &route_id);
std::string route_google_maps_url(const GpsTrace &trace);
