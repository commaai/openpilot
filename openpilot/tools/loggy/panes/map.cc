#include "tools/loggy/panes/map.h"

#include "tools/loggy/backend/session.h"
#include "tools/loggy/shell/theme.h"
#include "tools/loggy/shell/workspace.h"

#include "imgui.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace loggy {
namespace {

struct MapProjection {
  double min_x = 0.0;
  double min_y = 0.0;
  double scale = 1.0;
  ImVec2 origin;
  ImVec2 size;
  double cos_lat = 1.0;
};

double projected_x(double lon, double cos_lat) {
  return lon * cos_lat;
}

double projected_y(double lat) {
  return -lat;
}

MapProjection make_projection(const MapTrace &trace, ImVec2 origin, ImVec2 size) {
  const double center_lat = (trace.min_lat + trace.max_lat) * 0.5;
  const double cos_lat = std::max(0.20, std::cos(center_lat * M_PI / 180.0));
  const double min_x = projected_x(trace.min_lon, cos_lat);
  const double max_x = projected_x(trace.max_lon, cos_lat);
  const double min_y = projected_y(trace.max_lat);
  const double max_y = projected_y(trace.min_lat);
  const double span_x = std::max(max_x - min_x, 1.0e-6);
  const double span_y = std::max(max_y - min_y, 1.0e-6);
  const double scale = 0.88 * std::min(static_cast<double>(size.x) / span_x, static_cast<double>(size.y) / span_y);
  const double content_w = span_x * scale;
  const double content_h = span_y * scale;
  return {
    .min_x = min_x - ((static_cast<double>(size.x) - content_w) * 0.5 / scale),
    .min_y = min_y - ((static_cast<double>(size.y) - content_h) * 0.5 / scale),
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

}  // namespace

void draw_map_pane(Session &session, PaneInstance &pane) {
  const MapState state = parse_map_state(pane.state_json);
  TimeRange range = session.playback().route_range();
  if (!range.valid() || range.span() <= 0.0) range = session.view_range().range();
  const MapTrace trace = prepare_map_trace(session.store(), range, state, &session.timeline());

  ImGui::TextDisabled("%zu GPS points", trace.points.size());
  if (trace.valid()) {
    ImGui::SameLine();
    ImGui::TextDisabled("| %.5f, %.5f to %.5f, %.5f", trace.min_lat, trace.min_lon, trace.max_lat, trace.max_lon);
    if (trace.decimated) {
      ImGui::SameLine();
      ImGui::TextDisabled("| decimated");
    }
  }

  const ImVec2 avail = ImGui::GetContentRegionAvail();
  const ImVec2 size(std::max(1.0f, avail.x), std::max(1.0f, avail.y));
  const ImVec2 origin = ImGui::GetCursorScreenPos();
  ImDrawList *draw_list = ImGui::GetWindowDrawList();
  draw_map_grid(draw_list, origin, size);

  if (!trace.valid()) {
    const char *label = "No GPS trace in store";
    const ImVec2 text_size = ImGui::CalcTextSize(label);
    draw_list->AddText(ImVec2(origin.x + (size.x - text_size.x) * 0.5f,
                              origin.y + (size.y - text_size.y) * 0.5f),
                       ImGui::GetColorU32(ImGuiCol_TextDisabled), label);
    ImGui::Dummy(size);
    return;
  }

  const MapProjection projection = make_projection(trace, origin, size);
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

  const std::optional<MapTracePoint> tracker = map_trace_point_at_time(trace, session.playback().tracker_time());
  if (tracker.has_value()) {
    const ImVec2 marker = map_to_screen(projection, tracker->lat, tracker->lon);
    draw_car_marker(draw_list, marker, tracker->bearing_deg, ImGui::GetColorU32(color_rgb(238, 188, 82)));
  }

  ImGui::Dummy(size);
}

}  // namespace loggy
