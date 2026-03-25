#include "tools/jotpluggler/app_internal.h"

#include "implot.h"
#include "imgui_internal.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <set>

ImVec4 cabana_window_bg() {
  return color_rgb(53, 53, 53);
}

ImVec4 cabana_panel_bg() {
  return color_rgb(60, 63, 65);
}

ImVec4 cabana_panel_alt_bg() {
  return color_rgb(46, 47, 49);
}

ImVec4 cabana_border_color() {
  return color_rgb(77, 77, 77);
}

ImVec4 cabana_accent() {
  return color_rgb(47, 101, 202);
}

ImVec4 cabana_accent_hover() {
  return color_rgb(64, 120, 224);
}

ImVec4 cabana_accent_active() {
  return color_rgb(74, 132, 236);
}

ImVec4 cabana_muted_text() {
  return color_rgb(153, 153, 153);
}

void push_cabana_mode_style() {
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0.0f, 0.0f));
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(5.0f, 3.0f));
  ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(4.0f, 2.0f));
  ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 1.0f);
  ImGui::PushStyleVar(ImGuiStyleVar_GrabRounding, 1.0f);
  ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 0.0f);
  ImGui::PushStyleVar(ImGuiStyleVar_TabRounding, 0.0f);

  ImGui::PushStyleColor(ImGuiCol_WindowBg, cabana_window_bg());
  ImGui::PushStyleColor(ImGuiCol_ChildBg, cabana_panel_bg());
  ImGui::PushStyleColor(ImGuiCol_PopupBg, color_rgb(45, 45, 48));
  ImGui::PushStyleColor(ImGuiCol_Border, cabana_border_color());
  ImGui::PushStyleColor(ImGuiCol_Separator, cabana_border_color());
  ImGui::PushStyleColor(ImGuiCol_Text, color_rgb(187, 187, 187));
  ImGui::PushStyleColor(ImGuiCol_TextDisabled, cabana_muted_text());
  ImGui::PushStyleColor(ImGuiCol_FrameBg, color_rgb(41, 41, 43));
  ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, color_rgb(50, 53, 58));
  ImGui::PushStyleColor(ImGuiCol_FrameBgActive, color_rgb(58, 61, 66));
  ImGui::PushStyleColor(ImGuiCol_Button, cabana_panel_bg());
  ImGui::PushStyleColor(ImGuiCol_ButtonHovered, color_rgb(74, 78, 82));
  ImGui::PushStyleColor(ImGuiCol_ButtonActive, cabana_accent());
  ImGui::PushStyleColor(ImGuiCol_ScrollbarBg, color_rgb(45, 45, 48));
  ImGui::PushStyleColor(ImGuiCol_ScrollbarGrab, color_rgb(92, 96, 101));
  ImGui::PushStyleColor(ImGuiCol_ScrollbarGrabHovered, color_rgb(112, 118, 126));
  ImGui::PushStyleColor(ImGuiCol_ScrollbarGrabActive, color_rgb(132, 140, 150));
  ImGui::PushStyleColor(ImGuiCol_Header, cabana_accent());
  ImGui::PushStyleColor(ImGuiCol_HeaderHovered, cabana_accent_hover());
  ImGui::PushStyleColor(ImGuiCol_HeaderActive, cabana_accent_active());
  ImGui::PushStyleColor(ImGuiCol_Tab, color_rgb(56, 58, 61));
  ImGui::PushStyleColor(ImGuiCol_TabHovered, color_rgb(70, 74, 79));
  ImGui::PushStyleColor(ImGuiCol_TabSelected, color_rgb(67, 70, 74));
  ImGui::PushStyleColor(ImGuiCol_TabSelectedOverline, cabana_accent());
  ImGui::PushStyleColor(ImGuiCol_TabDimmed, color_rgb(49, 51, 54));
  ImGui::PushStyleColor(ImGuiCol_TabDimmedSelected, color_rgb(61, 64, 68));
  ImGui::PushStyleColor(ImGuiCol_TabDimmedSelectedOverline, cabana_accent());
  ImGui::PushStyleColor(ImGuiCol_TableHeaderBg, color_rgb(47, 47, 50));
  ImGui::PushStyleColor(ImGuiCol_TableBorderStrong, cabana_border_color());
  ImGui::PushStyleColor(ImGuiCol_TableBorderLight, color_rgb(69, 69, 72));
  ImGui::PushStyleColor(ImGuiCol_TableRowBgAlt, color_rgb(65, 68, 71, 0.35f));
}

void pop_cabana_mode_style() {
  ImGui::PopStyleColor(31);
  ImGui::PopStyleVar(8);
}

void draw_cabana_panel_title(const char *title, std::string_view subtitle) {
  app_push_bold_font();
  ImGui::TextUnformatted(title);
  app_pop_bold_font();
  if (!subtitle.empty()) {
    ImGui::SameLine();
    ImGui::TextDisabled("%.*s", static_cast<int>(subtitle.size()), subtitle.data());
  }
  ImGui::Spacing();
}

bool draw_cabana_bottom_tab(const char *id, const char *label, bool active, float width) {
  ImGui::PushStyleColor(ImGuiCol_Button, active ? cabana_window_bg() : cabana_panel_alt_bg());
  ImGui::PushStyleColor(ImGuiCol_ButtonHovered, active ? cabana_window_bg() : color_rgb(67, 70, 73));
  ImGui::PushStyleColor(ImGuiCol_ButtonActive, cabana_window_bg());
  const bool clicked = ImGui::Button((std::string(label) + id).c_str(), ImVec2(width, 26.0f));
  ImGui::PopStyleColor(3);
  const ImRect rect(ImGui::GetItemRectMin(), ImGui::GetItemRectMax());
  ImDrawList *draw = ImGui::GetWindowDrawList();
  draw->AddRect(rect.Min, rect.Max, ImGui::GetColorU32(active ? cabana_border_color() : color_rgb(92, 96, 101)));
  if (active) {
    draw->AddLine(ImVec2(rect.Min.x + 1.0f, rect.Max.y), ImVec2(rect.Max.x - 1.0f, rect.Max.y),
                  ImGui::GetColorU32(cabana_accent()), 2.0f);
  }
  return clicked;
}

void draw_cabana_detail_tab_strip(UiState *state) {
  const float strip_h = 30.0f;
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 4.0f));
  ImGui::PushStyleColor(ImGuiCol_ChildBg, cabana_panel_alt_bg());
  ImGui::BeginChild("##cabana_detail_bottom_tabs", ImVec2(0.0f, strip_h), false, ImGuiWindowFlags_NoScrollbar);
  const ImVec2 pos = ImGui::GetWindowPos();
  const ImVec2 size = ImGui::GetWindowSize();
  const ImRect rect(pos, ImVec2(pos.x + size.x, pos.y + size.y));
  ImDrawList *draw = ImGui::GetWindowDrawList();
  draw->AddLine(ImVec2(rect.Min.x, rect.Min.y + 1.0f), ImVec2(rect.Max.x, rect.Min.y + 1.0f),
                ImGui::GetColorU32(cabana_border_color()));
  ImGui::SetCursorPosX(8.0f);
  if (draw_cabana_bottom_tab("##msg", "Msg", state->cabana.detail_tab == 0, 72.0f)) {
    state->cabana.detail_tab = 0;
    state->cabana.detail_top_auto_fit = true;
  }
  ImGui::SameLine(0.0f, 4.0f);
  if (draw_cabana_bottom_tab("##logs", "Logs", state->cabana.detail_tab == 1, 76.0f)) {
    state->cabana.detail_tab = 1;
  }
  ImGui::EndChild();
  ImGui::PopStyleColor();
  ImGui::PopStyleVar();
}

void draw_cabana_welcome_panel() {
  const ImVec2 avail = ImGui::GetContentRegionAvail();
  const float center_x = ImGui::GetCursorPosX() + avail.x * 0.5f;
  ImGui::Dummy(ImVec2(0.0f, std::max(28.0f, avail.y * 0.18f)));
  app_push_bold_font();
  const char *title = "CABANA";
  const float title_w = ImGui::CalcTextSize(title).x;
  ImGui::SetCursorPosX(std::max(0.0f, center_x - title_w * 0.5f));
  ImGui::TextUnformatted(title);
  app_pop_bold_font();
  ImGui::Spacing();
  const char *hint = "<-Select a message to view details";
  const float hint_w = ImGui::CalcTextSize(hint).x;
  ImGui::SetCursorPosX(std::max(0.0f, center_x - hint_w * 0.5f));
  ImGui::TextDisabled("%s", hint);
}

namespace {

void draw_splitter_line(const ImRect &rect, bool hovered) {
  ImDrawList *draw_list = ImGui::GetWindowDrawList();
  const ImU32 color = hovered ? IM_COL32(112, 128, 144, 255) : IM_COL32(194, 198, 204, 255);
  if (rect.GetWidth() > rect.GetHeight()) {
    const float y = (rect.Min.y + rect.Max.y) * 0.5f;
    draw_list->AddLine(ImVec2(rect.Min.x, y), ImVec2(rect.Max.x, y), color, hovered ? 2.0f : 1.0f);
  } else {
    const float x = (rect.Min.x + rect.Max.x) * 0.5f;
    draw_list->AddLine(ImVec2(x, rect.Min.y), ImVec2(x, rect.Max.y), color, hovered ? 2.0f : 1.0f);
  }
}

}  // namespace

void draw_vertical_splitter(const char *id,
                            float height,
                            float min_left,
                            float max_left,
                            float *left_width) {
  const ImVec2 size(4.0f, height);
  ImGui::InvisibleButton(id, size);
  const bool hovered = ImGui::IsItemHovered() || ImGui::IsItemActive();
  if (hovered) {
    ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
  }
  if (ImGui::IsItemActive()) {
    *left_width = std::clamp(*left_width + ImGui::GetIO().MouseDelta.x, min_left, max_left);
  }
  draw_splitter_line(ImRect(ImGui::GetItemRectMin(), ImGui::GetItemRectMax()), hovered);
}

void draw_right_splitter(const char *id,
                         float height,
                         float min_right,
                         float max_right,
                         float *right_width) {
  const ImVec2 size(4.0f, height);
  ImGui::InvisibleButton(id, size);
  const bool hovered = ImGui::IsItemHovered() || ImGui::IsItemActive();
  if (hovered) {
    ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
  }
  if (ImGui::IsItemActive()) {
    *right_width = std::clamp(*right_width - ImGui::GetIO().MouseDelta.x, min_right, max_right);
  }
  draw_splitter_line(ImRect(ImGui::GetItemRectMin(), ImGui::GetItemRectMax()), hovered);
}

bool draw_horizontal_splitter(const char *id,
                              float width,
                              float min_top,
                              float max_top,
                              float *top_height) {
  const ImVec2 size(width, 4.0f);
  ImGui::InvisibleButton(id, size);
  const bool hovered = ImGui::IsItemHovered() || ImGui::IsItemActive();
  if (hovered) {
    ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
  }
  bool changed = false;
  if (ImGui::IsItemActive()) {
    *top_height = std::clamp(*top_height + ImGui::GetIO().MouseDelta.y, min_top, max_top);
    changed = true;
  }
  draw_splitter_line(ImRect(ImGui::GetItemRectMin(), ImGui::GetItemRectMax()), hovered);
  return changed;
}

void draw_payload_bytes(std::string_view data, const std::string *prev_data) {
  app_push_mono_font();
  for (size_t i = 0; i < data.size(); ++i) {
    if (i > 0) ImGui::SameLine(0.0f, 6.0f);
    const bool changed = prev_data != nullptr
                      && i < prev_data->size()
                      && static_cast<unsigned char>((*prev_data)[i]) != static_cast<unsigned char>(data[i]);
    if (changed) ImGui::PushStyleColor(ImGuiCol_Text, color_rgb(199, 74, 59));
    char hex[4];
    std::snprintf(hex, sizeof(hex), "%02X", static_cast<unsigned char>(data[i]));
    ImGui::TextUnformatted(hex);
    if (changed) ImGui::PopStyleColor();
  }
  app_pop_mono_font();
}

void draw_payload_preview_boxes(const char *id, std::string_view data, const std::string *prev_data, float max_width) {
  constexpr float kByteW = 17.0f;
  constexpr float kByteH = 16.0f;
  constexpr float kGap = 2.0f;
  const size_t capacity = std::max<size_t>(1, static_cast<size_t>((max_width + kGap) / (kByteW + kGap)));
  const size_t visible = std::min(data.size(), capacity);
  const bool truncated = visible < data.size();
  const float ellipsis_w = truncated ? 10.0f : 0.0f;
  const float width = std::max(18.0f, visible * (kByteW + kGap) - (visible > 0 ? kGap : 0.0f) + ellipsis_w);
  ImGui::InvisibleButton(id, ImVec2(width, kByteH));
  const ImRect rect(ImGui::GetItemRectMin(), ImGui::GetItemRectMax());
  ImDrawList *draw = ImGui::GetWindowDrawList();
  app_push_mono_font();
  for (size_t i = 0; i < visible; ++i) {
    const unsigned char after = static_cast<unsigned char>(data[i]);
    const bool has_prev = prev_data != nullptr && i < prev_data->size();
    const unsigned char before = has_prev ? static_cast<unsigned char>((*prev_data)[i]) : after;
    ImU32 fill = ImGui::GetColorU32(color_rgb(67, 70, 74));
    if (has_prev && after != before) {
      fill = ImGui::GetColorU32(after > before ? color_rgb(72, 95, 140) : color_rgb(120, 72, 68));
    }
    const float x0 = rect.Min.x + static_cast<float>(i) * (kByteW + kGap);
    const ImRect box(ImVec2(x0, rect.Min.y), ImVec2(x0 + kByteW, rect.Min.y + kByteH));
    draw->AddRectFilled(box.Min, box.Max, fill, 2.0f);
    draw->AddRect(box.Min, box.Max, ImGui::GetColorU32(color_rgb(105, 110, 116)), 2.0f);
    char hex[4];
    std::snprintf(hex, sizeof(hex), "%02X", after);
    const ImVec2 text_size = ImGui::CalcTextSize(hex);
    draw->AddText(ImGui::GetFont(),
                  ImGui::GetFontSize(),
                  ImVec2(box.Min.x + (box.GetWidth() - text_size.x) * 0.5f,
                         box.Min.y + (box.GetHeight() - text_size.y) * 0.5f - 1.0f),
                  ImGui::GetColorU32(color_rgb(228, 231, 236)),
                  hex);
  }
  if (truncated) {
    draw->AddText(ImVec2(rect.Max.x - 9.0f, rect.Min.y - 1.0f),
                  ImGui::GetColorU32(color_rgb(154, 160, 168)),
                  "...");
  }
  app_pop_mono_font();
}

void draw_signal_overlay_legend(const std::vector<std::pair<const CabanaSignalSummary *, ImU32>> &highlighted) {
  if (highlighted.empty()) {
    return;
  }
  app_push_bold_font();
  ImGui::TextUnformatted("Signals");
  app_pop_bold_font();
  for (size_t i = 0; i < highlighted.size(); ++i) {
    if (i > 0) ImGui::SameLine(0.0f, 12.0f);
    ImGui::ColorButton(("##cabana_signal_color_" + std::to_string(i)).c_str(),
                       ImGui::ColorConvertU32ToFloat4(highlighted[i].second),
                       ImGuiColorEditFlags_NoTooltip,
                       ImVec2(10.0f, 10.0f));
    ImGui::SameLine(0.0f, 6.0f);
    ImGui::TextUnformatted(highlighted[i].first->name.c_str());
    ImGui::SameLine(0.0f, 6.0f);
    ImGui::TextDisabled("[%d|%d]", highlighted[i].first->start_bit, highlighted[i].first->size);
  }
  ImGui::Spacing();
}

ImU32 mix_color(ImU32 a, ImU32 b, float t) {
  const ImVec4 av = ImGui::ColorConvertU32ToFloat4(a);
  const ImVec4 bv = ImGui::ColorConvertU32ToFloat4(b);
  return ImGui::GetColorU32(ImVec4(av.x + (bv.x - av.x) * t,
                                   av.y + (bv.y - av.y) * t,
                                   av.z + (bv.z - av.z) * t,
                                   av.w + (bv.w - av.w) * t));
}

void draw_empty_panel(const char *title, const char *message) {
  draw_cabana_panel_title(title);
  ImGui::TextDisabled("%s", message);
}

void draw_cabana_toolbar_button(const char *label, bool enabled, const std::function<void()> &on_click) {
  ImGui::BeginDisabled(!enabled);
  if (ImGui::Button(label)) {
    on_click();
  }
  ImGui::EndDisabled();
}

void draw_cabana_warning_banner(const std::vector<std::string> &warnings) {
  if (warnings.empty()) {
    return;
  }
  const float height = 28.0f + std::max(0.0f, (static_cast<float>(warnings.size()) - 1.0f) * 16.0f);
  ImGui::InvisibleButton("##cabana_warning_banner", ImVec2(ImGui::GetContentRegionAvail().x, height));
  const ImRect rect(ImGui::GetItemRectMin(), ImGui::GetItemRectMax());
  ImDrawList *draw = ImGui::GetWindowDrawList();
  draw->AddRectFilled(rect.Min, rect.Max, ImGui::GetColorU32(color_rgb(251, 245, 229)), 4.0f);
  draw->AddRect(rect.Min, rect.Max, ImGui::GetColorU32(color_rgb(221, 191, 121)), 4.0f, 0, 1.0f);
  draw->AddText(ImVec2(rect.Min.x + 10.0f, rect.Min.y + 6.0f),
                ImGui::GetColorU32(color_rgb(164, 106, 28)),
                "!");
  float y = rect.Min.y + 5.0f;
  for (const std::string &warning : warnings) {
    draw->AddText(ImVec2(rect.Min.x + 24.0f, y),
                  ImGui::GetColorU32(color_rgb(109, 82, 34)),
                  warning.c_str());
    y += 16.0f;
  }
}

void draw_signal_sparkline(const AppSession &session,
                           const UiState &state,
                           std::string_view signal_path,
                           bool selected,
                           ImVec2 size) {
  const RouteSeries *series = app_find_route_series(session, std::string(signal_path));
  if (size.x <= 0.0f) {
    size.x = std::max(96.0f, ImGui::GetContentRegionAvail().x);
  }
  if (size.y <= 0.0f) {
    size.y = 24.0f;
  }
  if (series == nullptr || series->times.size() < 2 || series->times.size() != series->values.size()) {
    ImGui::Dummy(size);
    return;
  }

  const std::string id = "##spark_" + std::string(signal_path);
  ImGui::InvisibleButton(id.c_str(), size);
  const ImRect rect(ImGui::GetItemRectMin(), ImGui::GetItemRectMax());
  ImDrawList *draw = ImGui::GetWindowDrawList();
  const ImU32 bg = ImGui::GetColorU32(selected ? color_rgb(59, 74, 103) : color_rgb(52, 54, 57));
  const ImU32 border = ImGui::GetColorU32(selected ? color_rgb(110, 145, 214) : color_rgb(93, 98, 104));
  const ImU32 line = ImGui::GetColorU32(selected ? color_rgb(109, 163, 255) : color_rgb(162, 170, 182));
  const ImU32 tracker = ImGui::GetColorU32(color_rgb(214, 93, 64));
  draw->AddRectFilled(rect.Min, rect.Max, bg, 4.0f);
  draw->AddRect(rect.Min, rect.Max, border, 4.0f);

  const double anchor = state.has_tracker_time
                      ? std::clamp(state.tracker_time, series->times.front(), series->times.back())
                      : series->times.back();
  double x_max = anchor;
  double x_min = std::max(series->times.front(), anchor - std::max(1, state.cabana.sparkline_range_sec));
  if (x_max <= x_min) {
    x_min = series->times.front();
    x_max = series->times.back();
  }
  if (x_max <= x_min) {
    return;
  }

  constexpr int kSamples = 40;
  std::array<double, kSamples> sampled = {};
  std::array<bool, kSamples> valid = {};
  bool found = false;
  double y_min = 0.0;
  double y_max = 0.0;
  for (int i = 0; i < kSamples; ++i) {
    const double t = x_min + (x_max - x_min) * static_cast<double>(i) / static_cast<double>(kSamples - 1);
    const std::optional<double> value = app_sample_xy_value_at_time(series->times, series->values, false, t);
    if (!value.has_value() || !std::isfinite(*value)) continue;
    sampled[static_cast<size_t>(i)] = *value;
    valid[static_cast<size_t>(i)] = true;
    if (!found) {
      y_min = y_max = *value;
      found = true;
    } else {
      y_min = std::min(y_min, *value);
      y_max = std::max(y_max, *value);
    }
  }
  if (!found) {
    return;
  }
  if (y_max <= y_min) {
    const double pad = std::max(0.1, std::abs(y_min) * 0.1);
    y_min -= pad;
    y_max += pad;
  } else {
    const double pad = (y_max - y_min) * 0.12;
    y_min -= pad;
    y_max += pad;
  }

  const float left = rect.Min.x + 4.0f;
  const float right = rect.Max.x - 4.0f;
  const float top = rect.Min.y + 4.0f;
  const float bottom = rect.Max.y - 4.0f;
  std::array<ImVec2, kSamples> points = {};
  int point_count = 0;
  for (int i = 0; i < kSamples; ++i) {
    if (!valid[static_cast<size_t>(i)]) {
      if (point_count > 1) draw->AddPolyline(points.data(), point_count, line, 0, selected ? 2.0f : 1.5f);
      point_count = 0;
      continue;
    }
    const float x = left + (right - left) * static_cast<float>(i) / static_cast<float>(kSamples - 1);
    const float frac = static_cast<float>((sampled[static_cast<size_t>(i)] - y_min) / (y_max - y_min));
    const float y = bottom - (bottom - top) * std::clamp(frac, 0.0f, 1.0f);
    points[static_cast<size_t>(point_count++)] = ImVec2(x, y);
  }
  if (point_count > 1) draw->AddPolyline(points.data(), point_count, line, 0, selected ? 2.0f : 1.5f);

  if (state.has_tracker_time && state.tracker_time >= x_min && state.tracker_time <= x_max) {
    const float marker_x = left + (right - left) * static_cast<float>((state.tracker_time - x_min) / (x_max - x_min));
    draw->AddLine(ImVec2(marker_x, top), ImVec2(marker_x, bottom), tracker, 1.5f);
  }
}

namespace {

struct CabanaChartSeries {
  const RouteSeries *series = nullptr;
  const SeriesFormat *format = nullptr;
  const EnumInfo *enum_info = nullptr;
  std::string path;
  std::string label;
  std::array<uint8_t, 3> color = {109, 163, 255};
};

const CabanaSignalSummary *find_message_signal(const CabanaMessageSummary &message, std::string_view path);

std::string cabana_chart_value_label(const AppSession &session, std::string_view path, double tracker_time) {
  const RouteSeries *series = app_find_route_series(session, std::string(path));
  if (series == nullptr || series->times.empty() || series->values.empty()) {
    return "--";
  }
  const auto value = app_sample_xy_value_at_time(series->times, series->values, false, tracker_time);
  const auto format_it = session.route_data.series_formats.find(std::string(path));
  const auto enum_it = session.route_data.enum_info.find(std::string(path));
  if (!value.has_value() || format_it == session.route_data.series_formats.end()) {
    return "--";
  }
  return format_display_value(*value,
                              format_it->second,
                              enum_it == session.route_data.enum_info.end() ? nullptr : &enum_it->second);
}

std::optional<std::pair<double, double>> current_chart_range(const UiState &state) {
  if (!state.has_shared_range || state.x_view_max <= state.x_view_min) {
    return std::nullopt;
  }
  return std::pair(state.x_view_min, state.x_view_max);
}

void apply_chart_range(UiState *state, std::optional<std::pair<double, double>> range) {
  if (!range.has_value()) {
    state->has_shared_range = true;
    state->x_view_min = state->route_x_min;
    state->x_view_max = std::max(state->route_x_min + MIN_HORIZONTAL_ZOOM_SECONDS, state->route_x_max);
  } else {
    state->has_shared_range = true;
    state->x_view_min = std::clamp(range->first, state->route_x_min, state->route_x_max);
    state->x_view_max = std::clamp(range->second, state->route_x_min, state->route_x_max);
    if (state->x_view_max - state->x_view_min < MIN_HORIZONTAL_ZOOM_SECONDS) {
      const double center = 0.5 * (state->x_view_min + state->x_view_max);
      state->x_view_min = std::max(state->route_x_min, center - 0.5 * MIN_HORIZONTAL_ZOOM_SECONDS);
      state->x_view_max = std::min(state->route_x_max, state->x_view_min + MIN_HORIZONTAL_ZOOM_SECONDS);
      if (state->x_view_max - state->x_view_min < MIN_HORIZONTAL_ZOOM_SECONDS) {
        state->x_view_min = std::max(state->route_x_min, state->route_x_max - MIN_HORIZONTAL_ZOOM_SECONDS);
        state->x_view_max = state->route_x_max;
      }
    }
  }
}

void push_chart_zoom_history(UiState *state) {
  const std::optional<std::pair<double, double>> range = current_chart_range(*state);
  if (!state->cabana.chart_zoom_history.empty() && state->cabana.chart_zoom_history.back() == range) {
    return;
  }
  state->cabana.chart_zoom_history.push_back(range);
  if (state->cabana.chart_zoom_history.size() > 50) {
    state->cabana.chart_zoom_history.erase(state->cabana.chart_zoom_history.begin());
  }
  state->cabana.chart_zoom_redo.clear();
}

void update_chart_range(UiState *state, double center, double width, bool push_history = true) {
  width = std::clamp(width, MIN_HORIZONTAL_ZOOM_SECONDS, std::max(MIN_HORIZONTAL_ZOOM_SECONDS, state->route_x_max - state->route_x_min));
  if (push_history) {
    push_chart_zoom_history(state);
  }
  std::pair<double, double> range(center - width * 0.5, center + width * 0.5);
  if (range.first < state->route_x_min) {
    range.second += state->route_x_min - range.first;
    range.first = state->route_x_min;
  }
  if (range.second > state->route_x_max) {
    range.first -= range.second - state->route_x_max;
    range.second = state->route_x_max;
  }
  if (range.first < state->route_x_min) {
    range.first = state->route_x_min;
  }
  apply_chart_range(state, range);
}

void reset_chart_range(UiState *state) {
  state->cabana.chart_zoom_history.clear();
  state->cabana.chart_zoom_redo.clear();
  apply_chart_range(state, std::nullopt);
}

CabanaChartTabState *active_chart_tab(UiState *state) {
  if (state->cabana.chart_tabs.empty()) {
    return nullptr;
  }
  state->cabana.active_chart_tab = std::clamp(state->cabana.active_chart_tab, 0,
                                              static_cast<int>(state->cabana.chart_tabs.size()) - 1);
  return &state->cabana.chart_tabs[static_cast<size_t>(state->cabana.active_chart_tab)];
}

void ensure_chart_tabs(UiState *state) {
  if (state->cabana.chart_tabs.empty()) {
    state->cabana.chart_tabs.push_back(CabanaChartTabState{.id = state->cabana.next_chart_tab_id++});
    state->cabana.active_chart_tab = 0;
    state->cabana.active_chart_index = 0;
  }
  state->cabana.active_chart_tab = std::clamp(state->cabana.active_chart_tab, 0,
                                              static_cast<int>(state->cabana.chart_tabs.size()) - 1);
  CabanaChartTabState &tab = state->cabana.chart_tabs[static_cast<size_t>(state->cabana.active_chart_tab)];
  state->cabana.active_chart_index = std::clamp(state->cabana.active_chart_index, 0,
                                                std::max(0, static_cast<int>(tab.charts.size()) - 1));
}

bool chart_has_signal(const CabanaChartState &chart, std::string_view path) {
  return std::find(chart.signal_paths.begin(), chart.signal_paths.end(), path) != chart.signal_paths.end();
}

CabanaChartState *ensure_active_chart(UiState *state) {
  ensure_chart_tabs(state);
  CabanaChartTabState *tab = active_chart_tab(state);
  if (tab == nullptr) {
    return nullptr;
  }
  if (tab->charts.empty()) {
    tab->charts.push_back(CabanaChartState{.id = state->cabana.next_chart_id++});
    state->cabana.active_chart_index = 0;
  }
  state->cabana.active_chart_index = std::clamp(state->cabana.active_chart_index, 0,
                                                static_cast<int>(tab->charts.size()) - 1);
  return &tab->charts[static_cast<size_t>(state->cabana.active_chart_index)];
}

void sync_chart_signal_aggregate(UiState *state) {
  std::set<std::string> ordered;
  for (const CabanaChartTabState &tab : state->cabana.chart_tabs) {
    for (const CabanaChartState &chart : tab.charts) {
      for (const std::string &path : chart.signal_paths) {
        ordered.insert(path);
      }
    }
  }
  state->cabana.chart_signal_paths.assign(ordered.begin(), ordered.end());
}

void reconcile_chart_tabs(UiState *state) {
  ensure_chart_tabs(state);
  std::set<std::string> desired(state->cabana.chart_signal_paths.begin(), state->cabana.chart_signal_paths.end());
  std::set<std::string> current;
  for (CabanaChartTabState &tab : state->cabana.chart_tabs) {
    for (CabanaChartState &chart : tab.charts) {
      chart.signal_paths.erase(std::remove_if(chart.signal_paths.begin(), chart.signal_paths.end(),
                                              [&](const std::string &path) { return !desired.count(path); }),
                               chart.signal_paths.end());
      chart.hidden.resize(chart.signal_paths.size(), false);
      current.insert(chart.signal_paths.begin(), chart.signal_paths.end());
    }
    tab.charts.erase(std::remove_if(tab.charts.begin(), tab.charts.end(), [](const CabanaChartState &chart) {
                      return chart.signal_paths.empty();
                    }),
                    tab.charts.end());
  }
  for (const std::string &path : desired) {
    if (current.count(path)) {
      continue;
    }
    CabanaChartState *chart = ensure_active_chart(state);
    if (chart != nullptr && !chart_has_signal(*chart, path)) {
      chart->signal_paths.push_back(path);
      chart->hidden.resize(chart->signal_paths.size(), false);
    }
  }
  ensure_chart_tabs(state);
  sync_chart_signal_aggregate(state);
}

double timeline_sec_from_mouse_x(double min_sec, double max_sec, float slider_x, float slider_w, float mouse_x) {
  if (slider_w <= 0.0f || max_sec <= min_sec) {
    return min_sec;
  }
  const float t = std::clamp((mouse_x - slider_x) / slider_w, 0.0f, 1.0f);
  return min_sec + (max_sec - min_sec) * t;
}

void draw_timeline_strip(ImDrawList *draw,
                         const ImVec2 &pos,
                         const ImVec2 &size,
                         const std::vector<TimelineEntry> &timeline,
                         double min_sec,
                         double max_sec,
                         double current_sec,
                         std::optional<std::pair<double, double>> highlight_range,
                         double hover_sec) {
  if (draw == nullptr || size.x <= 0.0f || size.y <= 0.0f || max_sec <= min_sec) {
    return;
  }
  const auto sec_to_x = [&](double sec) {
    const double t = (sec - min_sec) / std::max(0.001, max_sec - min_sec);
    return pos.x + static_cast<float>(t * size.x);
  };
  draw->AddRectFilled(pos, ImVec2(pos.x + size.x, pos.y + size.y),
                      ImGui::GetColorU32(color_rgb(54, 57, 60)));
  for (const TimelineEntry &entry : timeline) {
    const float x0 = sec_to_x(std::clamp(entry.start_time, min_sec, max_sec));
    const float x1 = sec_to_x(std::clamp(entry.end_time, min_sec, max_sec));
    if (x1 <= x0) {
      continue;
    }
    draw->AddRectFilled(ImVec2(x0, pos.y), ImVec2(x1, pos.y + size.y), timeline_entry_color(entry.type));
  }
  if (highlight_range.has_value()) {
    const float x0 = sec_to_x(std::clamp(highlight_range->first, min_sec, max_sec));
    const float x1 = sec_to_x(std::clamp(highlight_range->second, min_sec, max_sec));
    if (x1 > x0) {
      draw->AddRectFilled(ImVec2(x0, pos.y), ImVec2(x1, pos.y + size.y), IM_COL32(255, 255, 255, 24));
      draw->AddRect(ImVec2(x0, pos.y), ImVec2(x1, pos.y + size.y), IM_COL32(230, 230, 230, 140));
    }
  }
  if (hover_sec >= min_sec && hover_sec <= max_sec) {
    const float x = sec_to_x(hover_sec);
    draw->AddLine(ImVec2(x, pos.y), ImVec2(x, pos.y + size.y), IM_COL32(255, 204, 68, 180), 1.5f);
  }
  const float cursor_x = sec_to_x(std::clamp(current_sec, min_sec, max_sec));
  draw->AddLine(ImVec2(cursor_x, pos.y - 1.0f), ImVec2(cursor_x, pos.y + size.y + 1.0f), IM_COL32(255, 255, 255, 210), 2.0f);
}

}  // namespace

void draw_chart_panel(AppSession *session, UiState *state, const CabanaMessageSummary *message) {
  auto build_chart_series = [&](std::string_view path, size_t color_index) -> std::optional<CabanaChartSeries> {
    const RouteSeries *series = app_find_route_series(*session, std::string(path));
    if (series == nullptr || series->times.size() < 2 || series->times.size() != series->values.size()) {
      return std::nullopt;
    }
    static constexpr std::array<std::array<uint8_t, 3>, 8> kChartPalette = {{
      {109, 163, 255},
      {255, 122, 89},
      {92, 198, 131},
      {243, 191, 77},
      {176, 124, 255},
      {71, 191, 183},
      {234, 98, 98},
      {162, 170, 182},
    }};
    CabanaChartSeries out;
    out.series = series;
    out.path = std::string(path);
    out.color = kChartPalette[color_index % kChartPalette.size()];
    const size_t slash = out.path.find_last_of('/');
    out.label = slash == std::string::npos ? out.path : out.path.substr(slash + 1);
    auto format_it = session->route_data.series_formats.find(out.path);
    if (format_it != session->route_data.series_formats.end()) {
      out.format = &format_it->second;
    }
    auto enum_it = session->route_data.enum_info.find(out.path);
    if (enum_it != session->route_data.enum_info.end()) {
      out.enum_info = &enum_it->second;
    }
    return out;
  };

  auto visible_series_window = [&](const RouteSeries &series) {
    size_t begin_index = 0;
    size_t end_index = series.times.size();
    if (state->has_shared_range && state->x_view_max > state->x_view_min) {
      auto begin_it = std::lower_bound(series.times.begin(), series.times.end(), state->x_view_min);
      auto end_it = std::upper_bound(series.times.begin(), series.times.end(), state->x_view_max);
      begin_index = begin_it == series.times.begin() ? 0 : static_cast<size_t>(std::distance(series.times.begin(), begin_it - 1));
      end_index = end_it == series.times.end() ? series.times.size() : static_cast<size_t>(std::distance(series.times.begin(), end_it + 1));
      end_index = std::min(end_index, series.times.size());
    }
    return std::pair(begin_index, end_index);
  };

  auto visible_y_bounds = [&](const RouteSeries &series, size_t begin_index, size_t end_index) {
    double y_min = std::numeric_limits<double>::max();
    double y_max = std::numeric_limits<double>::lowest();
    for (size_t i = begin_index; i < end_index; ++i) {
      y_min = std::min(y_min, series.values[i]);
      y_max = std::max(y_max, series.values[i]);
    }
    if (y_min == std::numeric_limits<double>::max() || y_max == std::numeric_limits<double>::lowest()) {
      y_min = 0.0;
      y_max = 1.0;
    }
    if (y_max <= y_min) {
      const double pad = std::max(std::abs(y_min) * 0.1, 1.0);
      y_min -= pad;
      y_max += pad;
    } else {
      const double pad = std::max((y_max - y_min) * 0.08, 1.0e-3);
      y_min -= pad;
      y_max += pad;
    }
    return std::pair(y_min, y_max);
  };
  reconcile_chart_tabs(state);
  const CabanaSignalSummary *selected_signal = (message != nullptr && !state->cabana.selected_signal_path.empty())
    ? find_message_signal(*message, state->cabana.selected_signal_path)
    : nullptr;
  CabanaChartTabState *tab = active_chart_tab(state);
  const int active_series_type = (tab != nullptr && !tab->charts.empty())
    ? tab->charts[static_cast<size_t>(std::clamp(state->cabana.active_chart_index, 0, static_cast<int>(tab->charts.size()) - 1))].series_type
    : 0;
  const char *active_series_type_label = active_series_type == 1 ? "Step" : (active_series_type == 2 ? "Scatter" : "Line");

  ImGui::PushStyleColor(ImGuiCol_ChildBg, cabana_panel_alt_bg());
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(6.0f, 4.0f));
  ImGui::BeginChild("##cabana_charts_header", ImVec2(0.0f, 34.0f), false, ImGuiWindowFlags_NoScrollbar);
  app_push_bold_font();
  ImGui::Text("Charts: %zu", state->cabana.chart_signal_paths.size());
  app_pop_bold_font();
  ImGui::SameLine(0.0f, 8.0f);
  if (ImGui::SmallButton("New Chart")) {
    CabanaChartState chart{.id = state->cabana.next_chart_id++};
    if (selected_signal != nullptr) {
      chart.signal_paths.push_back(selected_signal->path);
      chart.hidden.push_back(false);
    }
    ensure_chart_tabs(state);
    active_chart_tab(state)->charts.push_back(std::move(chart));
    state->cabana.active_chart_index = static_cast<int>(active_chart_tab(state)->charts.size()) - 1;
    sync_chart_signal_aggregate(state);
  }
  ImGui::SameLine(0.0f, 4.0f);
  if (ImGui::SmallButton("New Tab")) {
    state->cabana.chart_tabs.push_back(CabanaChartTabState{.id = state->cabana.next_chart_tab_id++});
    state->cabana.active_chart_tab = static_cast<int>(state->cabana.chart_tabs.size()) - 1;
    state->cabana.active_chart_index = 0;
    if (selected_signal != nullptr) {
      CabanaChartState *chart = ensure_active_chart(state);
      if (chart != nullptr && !chart_has_signal(*chart, selected_signal->path)) {
        chart->signal_paths.push_back(selected_signal->path);
        chart->hidden.resize(chart->signal_paths.size(), false);
      }
    }
    sync_chart_signal_aggregate(state);
  }
  ImGui::SameLine(0.0f, 8.0f);
  if (ImGui::BeginCombo("##chart_type_header", active_series_type_label)) {
    if (ImGui::Selectable("Line", active_series_type == 0) && tab != nullptr && !tab->charts.empty()) {
      tab->charts[static_cast<size_t>(state->cabana.active_chart_index)].series_type = 0;
    }
    if (ImGui::Selectable("Step", active_series_type == 1) && tab != nullptr && !tab->charts.empty()) {
      tab->charts[static_cast<size_t>(state->cabana.active_chart_index)].series_type = 1;
    }
    if (ImGui::Selectable("Scatter", active_series_type == 2) && tab != nullptr && !tab->charts.empty()) {
      tab->charts[static_cast<size_t>(state->cabana.active_chart_index)].series_type = 2;
    }
    ImGui::EndCombo();
  }
  ImGui::SameLine(0.0f, 4.0f);
  if (ImGui::BeginCombo("##chart_cols_header", ("Columns: " + std::to_string(state->cabana.chart_columns)).c_str())) {
    for (int col = 1; col <= 3; ++col) {
      if (ImGui::Selectable(std::to_string(col).c_str(), state->cabana.chart_columns == col)) {
        state->cabana.chart_columns = col;
      }
    }
    ImGui::EndCombo();
  }
  ImGui::SameLine(0.0f, 6.0f);
  ImGui::BeginDisabled(state->cabana.chart_zoom_history.empty());
  if (ImGui::SmallButton("Undo Zoom")) {
    state->cabana.chart_zoom_redo.push_back(current_chart_range(*state));
    apply_chart_range(state, state->cabana.chart_zoom_history.back());
    state->cabana.chart_zoom_history.pop_back();
  }
  ImGui::EndDisabled();
  ImGui::SameLine(0.0f, 6.0f);
  ImGui::BeginDisabled(state->cabana.chart_zoom_redo.empty());
  if (ImGui::SmallButton("Redo Zoom")) {
    state->cabana.chart_zoom_history.push_back(current_chart_range(*state));
    apply_chart_range(state, state->cabana.chart_zoom_redo.back());
    state->cabana.chart_zoom_redo.pop_back();
  }
  ImGui::EndDisabled();
  ImGui::SameLine(0.0f, 6.0f);
  if (ImGui::SmallButton("Reset")) {
    reset_chart_range(state);
  }
  ImGui::SameLine(0.0f, 6.0f);
  ImGui::BeginDisabled(selected_signal == nullptr);
  if (ImGui::SmallButton("Add Signal")) {
    CabanaChartState *chart = ensure_active_chart(state);
    if (chart != nullptr && selected_signal != nullptr && !chart_has_signal(*chart, selected_signal->path)) {
      chart->signal_paths.push_back(selected_signal->path);
      chart->hidden.resize(chart->signal_paths.size(), false);
      sync_chart_signal_aggregate(state);
    }
  }
  ImGui::EndDisabled();
  ImGui::SameLine(0.0f, 6.0f);
  if (ImGui::SmallButton("Remove All")) {
    if (tab != nullptr) {
      tab->charts.clear();
    }
    sync_chart_signal_aggregate(state);
  }
  ImGui::SameLine(0.0f, 8.0f);
  const auto range = current_chart_range(*state);
  if (range.has_value()) {
    ImGui::TextDisabled("%.2f - %.2f", range->first, range->second);
  } else {
    ImGui::TextDisabled("full route");
  }
  ImGui::EndChild();
  ImGui::PopStyleVar();
  ImGui::PopStyleColor();

  push_cabana_mode_style();
  if (state->cabana.chart_tabs.size() > 1 && ImGui::BeginTabBar("##cabana_chart_tabs", ImGuiTabBarFlags_FittingPolicyResizeDown | ImGuiTabBarFlags_Reorderable)) {
    int remove_tab = -1;
    int duplicate_tab = -1;
    int close_other_tab = -1;
    for (int i = 0; i < static_cast<int>(state->cabana.chart_tabs.size()); ++i) {
      bool open = true;
      const std::string label = "Tab " + std::to_string(i + 1) + " (" + std::to_string(state->cabana.chart_tabs[static_cast<size_t>(i)].charts.size()) + ")";
      const ImGuiTabItemFlags flags = state->cabana.active_chart_tab == i ? ImGuiTabItemFlags_SetSelected : 0;
      if (ImGui::BeginTabItem(label.c_str(), &open, flags)) {
        state->cabana.active_chart_tab = i;
        ImGui::EndTabItem();
      }
      if (ImGui::BeginPopupContextItem(("##chart_tab_ctx" + std::to_string(state->cabana.chart_tabs[static_cast<size_t>(i)].id)).c_str())) {
        if (ImGui::MenuItem("Duplicate Tab")) duplicate_tab = i;
        if (ImGui::MenuItem("Close Other Tabs", nullptr, false, state->cabana.chart_tabs.size() > 1)) close_other_tab = i;
        if (ImGui::MenuItem("Close Tab", nullptr, false, state->cabana.chart_tabs.size() > 1)) remove_tab = i;
        ImGui::EndPopup();
      }
      if (!open && state->cabana.chart_tabs.size() > 1) remove_tab = i;
    }
    if (duplicate_tab >= 0) {
      CabanaChartTabState dup = state->cabana.chart_tabs[static_cast<size_t>(duplicate_tab)];
      dup.id = state->cabana.next_chart_tab_id++;
      for (CabanaChartState &chart : dup.charts) {
        chart.id = state->cabana.next_chart_id++;
      }
      state->cabana.chart_tabs.insert(state->cabana.chart_tabs.begin() + duplicate_tab + 1, std::move(dup));
      state->cabana.active_chart_tab = duplicate_tab + 1;
    }
    if (close_other_tab >= 0) {
      CabanaChartTabState keep = std::move(state->cabana.chart_tabs[static_cast<size_t>(close_other_tab)]);
      state->cabana.chart_tabs.assign(1, std::move(keep));
      state->cabana.active_chart_tab = 0;
    }
    if (remove_tab >= 0 && state->cabana.chart_tabs.size() > 1) {
      state->cabana.chart_tabs.erase(state->cabana.chart_tabs.begin() + remove_tab);
      state->cabana.active_chart_tab = std::clamp(state->cabana.active_chart_tab, 0,
                                                  static_cast<int>(state->cabana.chart_tabs.size()) - 1);
    }
    ImGui::EndTabBar();
  }
  pop_cabana_mode_style();

  tab = active_chart_tab(state);
  const auto display_range = current_chart_range(*state).value_or(std::pair(state->route_x_min, state->route_x_max));
  double x_min = display_range.first;
  double x_max = display_range.second;

  const ImVec2 timeline_pos = ImGui::GetCursorScreenPos();
  const ImVec2 timeline_size(ImGui::GetContentRegionAvail().x, 14.0f);
  ImGui::InvisibleButton("##cabana_chart_timeline", timeline_size);
  const bool timeline_hovered = ImGui::IsItemHovered();
  if (timeline_hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
    state->cabana.chart_timeline_zoom_drag_active = true;
    state->cabana.chart_timeline_zoom_start_x = std::clamp(ImGui::GetIO().MousePos.x, timeline_pos.x, timeline_pos.x + timeline_size.x);
    state->cabana.chart_timeline_zoom_min_x = timeline_pos.x;
    state->cabana.chart_timeline_zoom_max_x = timeline_pos.x + timeline_size.x;
    state->cabana.chart_timeline_zoom_range_min = x_min;
    state->cabana.chart_timeline_zoom_range_max = x_max;
  }
  const double timeline_hover_sec = (timeline_hovered || state->cabana.chart_timeline_zoom_drag_active)
    ? timeline_sec_from_mouse_x(x_min, x_max, timeline_pos.x, timeline_size.x, ImGui::GetIO().MousePos.x)
    : -1.0;
  draw_timeline_strip(ImGui::GetWindowDrawList(),
                      timeline_pos,
                      timeline_size,
                      session->route_data.timeline,
                      x_min,
                      x_max,
                      state->tracker_time,
                      state->cabana.chart_timeline_zoom_drag_active
                        ? std::optional<std::pair<double, double>>(std::pair(
                            std::min(timeline_sec_from_mouse_x(state->cabana.chart_timeline_zoom_range_min, state->cabana.chart_timeline_zoom_range_max,
                                                                timeline_pos.x, timeline_size.x, state->cabana.chart_timeline_zoom_start_x),
                                     timeline_hover_sec),
                            std::max(timeline_sec_from_mouse_x(state->cabana.chart_timeline_zoom_range_min, state->cabana.chart_timeline_zoom_range_max,
                                                                timeline_pos.x, timeline_size.x, state->cabana.chart_timeline_zoom_start_x),
                                     timeline_hover_sec)))
                        : std::nullopt,
                      timeline_hover_sec >= 0 ? timeline_hover_sec : state->cabana.chart_hover_sec);
  if (state->cabana.chart_timeline_zoom_drag_active && ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
    const float current_x = std::clamp(ImGui::GetIO().MousePos.x, state->cabana.chart_timeline_zoom_min_x, state->cabana.chart_timeline_zoom_max_x);
    const float drag_px = std::abs(current_x - state->cabana.chart_timeline_zoom_start_x);
    if (drag_px > 6.0f) {
      const double zoom_min = timeline_sec_from_mouse_x(state->cabana.chart_timeline_zoom_range_min,
                                                        state->cabana.chart_timeline_zoom_range_max,
                                                        state->cabana.chart_timeline_zoom_min_x,
                                                        state->cabana.chart_timeline_zoom_max_x - state->cabana.chart_timeline_zoom_min_x,
                                                        std::min(current_x, state->cabana.chart_timeline_zoom_start_x));
      const double zoom_max = timeline_sec_from_mouse_x(state->cabana.chart_timeline_zoom_range_min,
                                                        state->cabana.chart_timeline_zoom_range_max,
                                                        state->cabana.chart_timeline_zoom_min_x,
                                                        state->cabana.chart_timeline_zoom_max_x - state->cabana.chart_timeline_zoom_min_x,
                                                        std::max(current_x, state->cabana.chart_timeline_zoom_start_x));
      if (zoom_max - zoom_min > MIN_HORIZONTAL_ZOOM_SECONDS) {
        update_chart_range(state, 0.5 * (zoom_min + zoom_max), zoom_max - zoom_min);
      }
    } else if (timeline_hover_sec >= 0.0) {
      state->tracker_time = std::clamp(timeline_hover_sec, state->route_x_min, state->route_x_max);
      state->has_tracker_time = true;
    }
    state->cabana.chart_timeline_zoom_drag_active = false;
  }
  ImGui::Dummy(ImVec2(0.0f, 4.0f));

  ImGui::PushStyleColor(ImGuiCol_ChildBg, cabana_window_bg());
  ImGui::BeginChild("##cabana_chart_plot", ImVec2(0.0f, 0.0f), false, ImGuiWindowFlags_AlwaysVerticalScrollbar);
  if (tab == nullptr || tab->charts.empty()) {
    ImGui::TextDisabled("No charts. Use 'New Chart' or 'Add Signal'.");
    ImGui::EndChild();
    ImGui::PopStyleColor();
    sync_chart_signal_aggregate(state);
    return;
  }

  int remove_chart_idx = -1;
  int split_chart_idx = -1;
  int drag_src_idx = -1;
  int drag_dst_idx = -1;
  bool drag_insert_after = false;
  double hover_sec_this_frame = -1.0;
  const int chart_count = static_cast<int>(tab->charts.size());
  const int eff_columns = std::min(std::clamp(state->cabana.chart_columns, 1, 3), std::max(1, chart_count));
  const int rows = (chart_count + eff_columns - 1) / eff_columns;
  const float gap = 4.0f;
  const float cell_w = std::max(220.0f, (ImGui::GetContentRegionAvail().x - gap * (eff_columns - 1)) / eff_columns);
  const float cell_h = std::max(140.0f, (ImGui::GetContentRegionAvail().y - gap * std::max(0, rows - 1)) / std::max(1, rows));

  for (int ci = 0; ci < chart_count; ++ci) {
    CabanaChartState &chart = tab->charts[static_cast<size_t>(ci)];
    if ((ci % eff_columns) != 0) {
      ImGui::SameLine(0.0f, gap);
    }
    std::vector<CabanaChartSeries> series_entries;
    series_entries.reserve(chart.signal_paths.size());
    for (size_t i = 0; i < chart.signal_paths.size(); ++i) {
      if (auto entry = build_chart_series(chart.signal_paths[i], i); entry.has_value()) {
        series_entries.push_back(std::move(*entry));
      }
    }
    if (static_cast<int>(chart.hidden.size()) < static_cast<int>(chart.signal_paths.size())) {
      chart.hidden.resize(chart.signal_paths.size(), false);
    }

    ImGui::PushID(chart.id);
    ImGui::BeginChild("##cabana_chart_cell", ImVec2(cell_w, cell_h), true);
    if (ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByPopup)) {
      state->cabana.active_chart_index = ci;
    }
    ImGui::BeginGroup();
    const std::string drag_id = "##chart_drag_" + std::to_string(chart.id);
    ImGui::SmallButton(drag_id.c_str());
    if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID)) {
      ImGui::SetDragDropPayload("CABANA_CHART", &ci, sizeof(int));
      ImGui::TextUnformatted("Drag chart");
      ImGui::TextDisabled("Drop onto a chart to merge");
      ImGui::TextDisabled("Hold Shift to reorder");
      ImGui::EndDragDropSource();
    }
    ImGui::SameLine(0.0f, 6.0f);
    for (size_t si = 0; si < series_entries.size(); ++si) {
      if (si > 0) ImGui::SameLine();
      const bool hidden = si < chart.hidden.size() && chart.hidden[si];
      if (!hidden) {
        ImGui::TextColored(color_rgb(series_entries[si].color), "%s", series_entries[si].label.c_str());
      } else {
        ImGui::TextDisabled("[%s]", series_entries[si].label.c_str());
      }
      if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && si < chart.hidden.size()) {
        chart.hidden[si] = !chart.hidden[si];
      }
      ImGui::SameLine(0.0f, 4.0f);
      ImGui::TextDisabled("%s", cabana_chart_value_label(*session, series_entries[si].path, state->tracker_time).c_str());
    }
    ImGui::SameLine(std::max(ImGui::GetCursorPosX() + 6.0f, ImGui::GetWindowContentRegionMax().x - 90.0f));
    if (ImGui::SmallButton("Type")) {
      ImGui::OpenPopup("##chart_type_popup");
    }
    ImGui::SameLine(0.0f, 4.0f);
    if (ImGui::SmallButton("x")) {
      remove_chart_idx = ci;
    }
    if (ImGui::BeginPopup("##chart_type_popup")) {
      if (ImGui::MenuItem("Line", nullptr, chart.series_type == 0)) chart.series_type = 0;
      if (ImGui::MenuItem("Step", nullptr, chart.series_type == 1)) chart.series_type = 1;
      if (ImGui::MenuItem("Scatter", nullptr, chart.series_type == 2)) chart.series_type = 2;
      ImGui::Separator();
      if (series_entries.size() > 1 && ImGui::MenuItem("Split Chart")) split_chart_idx = ci;
      if (ImGui::MenuItem("Close Chart")) remove_chart_idx = ci;
      ImGui::Separator();
      if (ImGui::MenuItem("Undo Zoom", nullptr, false, !state->cabana.chart_zoom_history.empty())) {
        state->cabana.chart_zoom_redo.push_back(current_chart_range(*state));
        apply_chart_range(state, state->cabana.chart_zoom_history.back());
        state->cabana.chart_zoom_history.pop_back();
      }
      if (ImGui::MenuItem("Redo Zoom", nullptr, false, !state->cabana.chart_zoom_redo.empty())) {
        state->cabana.chart_zoom_history.push_back(current_chart_range(*state));
        apply_chart_range(state, state->cabana.chart_zoom_redo.back());
        state->cabana.chart_zoom_redo.pop_back();
      }
      if (ImGui::MenuItem("Reset Zoom")) {
        reset_chart_range(state);
      }
      ImGui::EndPopup();
    }
    ImGui::EndGroup();

    const ImVec2 plot_size(ImGui::GetContentRegionAvail().x, std::max(96.0f, ImGui::GetContentRegionAvail().y - 2.0f));
    ImGui::PushStyleColor(ImGuiCol_FrameBg, color_rgb(52, 54, 57));
    ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, color_rgb(60, 63, 66));
    ImGui::PushStyleColor(ImGuiCol_FrameBgActive, color_rgb(67, 70, 74));
    ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(8.0f, 6.0f));
    ImPlot::PushStyleVar(ImPlotStyleVar_LabelPadding, ImVec2(5.0f, 2.0f));
    ImPlot::PushStyleColor(ImPlotCol_PlotBg, color_rgb(52, 54, 57));
    ImPlot::PushStyleColor(ImPlotCol_PlotBorder, color_rgb(95, 100, 106));
    ImPlot::PushStyleColor(ImPlotCol_LegendBg, color_rgb(46, 47, 49, 0.94f));
    ImPlot::PushStyleColor(ImPlotCol_LegendBorder, color_rgb(95, 100, 106));
    ImPlot::PushStyleColor(ImPlotCol_LegendText, color_rgb(220, 224, 229));
    ImPlot::PushStyleColor(ImPlotCol_TitleText, color_rgb(220, 224, 229));
    ImPlot::PushStyleColor(ImPlotCol_InlayText, color_rgb(214, 219, 225));
    ImPlot::PushStyleColor(ImPlotCol_AxisGrid, color_rgb(86, 90, 96));
    ImPlot::PushStyleColor(ImPlotCol_AxisText, color_rgb(182, 188, 196));
    ImPlot::PushStyleColor(ImPlotCol_AxisBg, color_rgb(47, 49, 52));
    ImPlot::PushStyleColor(ImPlotCol_AxisBgHovered, color_rgb(52, 54, 57, 0.96f));
    ImPlot::PushStyleColor(ImPlotCol_AxisBgActive, color_rgb(58, 61, 65, 0.98f));
    ImPlot::PushStyleColor(ImPlotCol_Selection, color_rgb(117, 161, 242, 0.22f));
    ImPlot::PushStyleColor(ImPlotCol_Crosshairs, color_rgb(214, 219, 225, 0.70f));

    if (ImPlot::BeginPlot("##cabana_signal_plot", plot_size, ImPlotFlags_NoMenus | ImPlotFlags_NoBoxSelect | ImPlotFlags_NoMouseText | ImPlotFlags_NoLegend)) {
      ImPlotAxisFlags x_flags = rows > 1 && ci < chart_count - eff_columns ? ImPlotAxisFlags_NoTickLabels : ImPlotAxisFlags_None;
      ImPlot::SetupAxes(ci >= chart_count - eff_columns ? "Time (s)" : nullptr, nullptr,
                        x_flags | ImPlotAxisFlags_NoMenus | ImPlotAxisFlags_NoHighlight,
                        ImPlotAxisFlags_NoMenus | ImPlotAxisFlags_NoHighlight);
      ImPlot::SetupAxisLinks(ImAxis_X1, &state->x_view_min, &state->x_view_max);
      if (state->route_x_max > state->route_x_min) {
        ImPlot::SetupAxisLimitsConstraints(ImAxis_X1, state->route_x_min, state->route_x_max);
      }

      double local_min = std::numeric_limits<double>::max();
      double local_max = std::numeric_limits<double>::lowest();
      for (size_t si = 0; si < series_entries.size(); ++si) {
        if (si < chart.hidden.size() && chart.hidden[si]) continue;
        const auto [begin_index, end_index] = visible_series_window(*series_entries[si].series);
        const auto [y_min, y_max] = visible_y_bounds(*series_entries[si].series, begin_index, end_index);
        local_min = std::min(local_min, y_min);
        local_max = std::max(local_max, y_max);
      }
      if (local_min == std::numeric_limits<double>::max() || local_max == std::numeric_limits<double>::lowest()) {
        local_min = -1.0;
        local_max = 1.0;
      }
      ImPlot::SetupAxisLimits(ImAxis_Y1, local_min, local_max, ImPlotCond_Always);

      const bool plot_hovered = ImPlot::IsPlotHovered();
      const double hover_sec = plot_hovered ? ImPlot::GetPlotMousePos().x : state->cabana.chart_hover_sec;
      if (plot_hovered) hover_sec_this_frame = hover_sec;

      for (size_t si = 0; si < series_entries.size(); ++si) {
        const CabanaChartSeries &entry = series_entries[si];
        if (si < chart.hidden.size() && chart.hidden[si]) continue;
        const auto [begin_index, end_index] = visible_series_window(*entry.series);
        if (end_index <= begin_index + 1) continue;
        ImPlotSpec spec;
        spec.LineColor = color_rgb(entry.color);
        spec.LineWeight = 2.0f;
        const int count = static_cast<int>(end_index - begin_index);
        const double *xs = entry.series->times.data() + begin_index;
        const double *ys = entry.series->values.data() + begin_index;
        const std::string legend_label = entry.label + "##" + entry.path;
        if (chart.series_type == 1) {
          spec.Flags = ImPlotStairsFlags_PreStep;
          ImPlot::PlotStairs(legend_label.c_str(), xs, ys, count, spec);
        } else if (chart.series_type == 2) {
          spec.Flags = ImPlotScatterFlags_None;
          ImPlot::PlotScatter(legend_label.c_str(), xs, ys, count, spec);
        } else {
          spec.Flags = ImPlotLineFlags_SkipNaN;
          ImPlot::PlotLine(legend_label.c_str(), xs, ys, count, spec);
        }

        if (hover_sec >= state->route_x_min && hover_sec <= state->route_x_max) {
          auto it = std::upper_bound(entry.series->times.begin(), entry.series->times.end(), hover_sec);
          const int idx = (it == entry.series->times.begin()) ? 0 : static_cast<int>(it - entry.series->times.begin()) - 1;
          if (idx >= 0 && idx < static_cast<int>(entry.series->times.size())) {
            const ImVec2 pos = ImPlot::PlotToPixels(entry.series->times[static_cast<size_t>(idx)], entry.series->values[static_cast<size_t>(idx)]);
            ImDrawList *draw = ImPlot::GetPlotDrawList();
            draw->AddCircleFilled(pos, 4.5f, ImGui::GetColorU32(color_rgb(entry.color)));
            draw->AddCircle(pos, 4.5f, IM_COL32(255, 255, 255, 180), 0, 1.2f);
          }
        }
      }

      if (state->has_tracker_time) {
        const double clamped = std::clamp(state->tracker_time, state->route_x_min, state->route_x_max);
        ImPlotSpec cursor_spec;
        cursor_spec.LineColor = color_rgb(214, 219, 225, 0.7f);
        cursor_spec.LineWeight = 1.0f;
        cursor_spec.Flags = ImPlotItemFlags_NoLegend;
        ImPlot::PlotInfLines("##tracker_cursor", &clamped, 1, cursor_spec);
      }

      if (plot_hovered) {
        if (ImGui::GetIO().MouseWheel != 0.0f) {
          const double center = std::clamp(hover_sec, state->route_x_min, state->route_x_max);
          const double width = std::clamp((state->x_view_max - state->x_view_min) * (ImGui::GetIO().MouseWheel > 0.0f ? 0.8 : 1.25),
                                          MIN_HORIZONTAL_ZOOM_SECONDS,
                                          std::max(MIN_HORIZONTAL_ZOOM_SECONDS, state->route_x_max - state->route_x_min));
          update_chart_range(state, center, width);
        }
        if (ImGui::IsMouseDragging(ImGuiMouseButton_Middle)) {
          const ImVec2 delta_px = ImGui::GetIO().MouseDelta;
          if (std::abs(delta_px.x) > 0.1f) {
            const ImPlotRect limits = ImPlot::GetPlotLimits();
            const double pps = ImPlot::GetPlotSize().x / (limits.X.Max - limits.X.Min);
            update_chart_range(state, 0.5 * (state->x_view_min + state->x_view_max) - delta_px.x / pps,
                               state->x_view_max - state->x_view_min, false);
          }
        }
        if (ImGui::IsMouseDragging(ImGuiMouseButton_Left) && ImGui::GetIO().KeyShift) {
          if (!state->cabana.chart_scrub_was_playing && state->playback_playing) {
            state->cabana.chart_scrub_was_playing = true;
            state->playback_playing = false;
          }
          state->tracker_time = std::clamp(hover_sec, state->route_x_min, state->route_x_max);
          state->has_tracker_time = true;
        }
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !ImGui::GetIO().KeyShift) {
          state->cabana.chart_zoom_drag_active = true;
          state->cabana.chart_zoom_drag_chart_id = chart.id;
          const ImVec2 plot_pos = ImPlot::GetPlotPos();
          const ImVec2 plot_sz = ImPlot::GetPlotSize();
          state->cabana.chart_zoom_drag_plot_min_x = plot_pos.x;
          state->cabana.chart_zoom_drag_plot_min_y = plot_pos.y;
          state->cabana.chart_zoom_drag_plot_max_x = plot_pos.x + plot_sz.x;
          state->cabana.chart_zoom_drag_plot_max_y = plot_pos.y + plot_sz.y;
          state->cabana.chart_zoom_drag_start_x = std::clamp(ImGui::GetIO().MousePos.x, plot_pos.x, plot_pos.x + plot_sz.x);
        }
      }
      if (state->cabana.chart_zoom_drag_active && state->cabana.chart_zoom_drag_chart_id == chart.id) {
        const float cur_x = std::clamp(ImGui::GetIO().MousePos.x,
                                       state->cabana.chart_zoom_drag_plot_min_x,
                                       state->cabana.chart_zoom_drag_plot_max_x);
        const float drag_px = std::abs(cur_x - state->cabana.chart_zoom_drag_start_x);
        if (ImGui::IsMouseDown(ImGuiMouseButton_Left) && !ImGui::GetIO().KeyShift && drag_px > 6.0f) {
          ImDrawList *overlay = ImPlot::GetPlotDrawList();
          const float sel_min_x = std::min(cur_x, state->cabana.chart_zoom_drag_start_x);
          const float sel_max_x = std::max(cur_x, state->cabana.chart_zoom_drag_start_x);
          overlay->AddRectFilled(ImVec2(sel_min_x, state->cabana.chart_zoom_drag_plot_min_y),
                                 ImVec2(sel_max_x, state->cabana.chart_zoom_drag_plot_max_y),
                                 IM_COL32(180, 205, 230, 40));
          overlay->AddRect(ImVec2(sel_min_x, state->cabana.chart_zoom_drag_plot_min_y),
                           ImVec2(sel_max_x, state->cabana.chart_zoom_drag_plot_max_y),
                           IM_COL32(180, 205, 230, 180), 0.0f, 0, 1.0f);
        }
        if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
          if (!ImGui::GetIO().KeyShift && drag_px > 6.0f) {
            const double min_x = std::clamp(ImPlot::PixelsToPlot(ImVec2(std::min(cur_x, state->cabana.chart_zoom_drag_start_x),
                                                                         state->cabana.chart_zoom_drag_plot_min_y)).x,
                                            state->route_x_min, state->route_x_max);
            const double max_x = std::clamp(ImPlot::PixelsToPlot(ImVec2(std::max(cur_x, state->cabana.chart_zoom_drag_start_x),
                                                                         state->cabana.chart_zoom_drag_plot_min_y)).x,
                                            state->route_x_min, state->route_x_max);
            if (max_x - min_x > MIN_HORIZONTAL_ZOOM_SECONDS) {
              update_chart_range(state, 0.5 * (min_x + max_x), max_x - min_x);
            }
          } else if (!ImGui::GetIO().KeyShift && plot_hovered) {
            state->tracker_time = std::clamp(hover_sec, state->route_x_min, state->route_x_max);
            state->has_tracker_time = true;
          }
          state->cabana.chart_zoom_drag_active = false;
          state->cabana.chart_zoom_drag_chart_id = -1;
        }
      }

      if (ImGui::BeginPopupContextWindow("##chart_ctx")) {
        if (ImGui::MenuItem("Line", nullptr, chart.series_type == 0)) chart.series_type = 0;
        if (ImGui::MenuItem("Step", nullptr, chart.series_type == 1)) chart.series_type = 1;
        if (ImGui::MenuItem("Scatter", nullptr, chart.series_type == 2)) chart.series_type = 2;
        ImGui::Separator();
        if (series_entries.size() > 1 && ImGui::MenuItem("Split Chart")) split_chart_idx = ci;
        if (ImGui::MenuItem("Close Chart")) remove_chart_idx = ci;
        ImGui::EndPopup();
      }
      ImPlot::EndPlot();
    }
    ImPlot::PopStyleColor(12);
    ImPlot::PopStyleVar(2);
    ImGui::PopStyleColor(3);
    ImGui::EndChild();

    if (ImGui::BeginDragDropTarget()) {
      if (const ImGuiPayload *payload = ImGui::AcceptDragDropPayload("CABANA_CHART")) {
        drag_src_idx = *static_cast<const int *>(payload->Data);
        drag_dst_idx = ci;
        const ImRect rect(ImGui::GetItemRectMin(), ImGui::GetItemRectMax());
        drag_insert_after = ImGui::GetMousePos().y >= (rect.Min.y + rect.Max.y) * 0.5f;
      }
      ImGui::EndDragDropTarget();
    }
    ImGui::PopID();
  }

  if (state->cabana.chart_scrub_was_playing && ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
    state->playback_playing = true;
    state->cabana.chart_scrub_was_playing = false;
  }
  state->cabana.chart_hover_sec = hover_sec_this_frame >= 0.0 ? hover_sec_this_frame : state->cabana.chart_hover_sec;

  if (split_chart_idx >= 0 && split_chart_idx < static_cast<int>(tab->charts.size())) {
    CabanaChartState &src = tab->charts[static_cast<size_t>(split_chart_idx)];
    if (src.signal_paths.size() > 1) {
      int pos = split_chart_idx + 1;
      for (size_t si = 1; si < src.signal_paths.size(); ++si) {
        CabanaChartState chart_copy{.id = state->cabana.next_chart_id++};
        chart_copy.series_type = src.series_type;
        chart_copy.signal_paths = {src.signal_paths[si]};
        chart_copy.hidden = {false};
        tab->charts.insert(tab->charts.begin() + pos, std::move(chart_copy));
        ++pos;
      }
      src.signal_paths.resize(1);
      src.hidden.resize(1, false);
    }
  }
  if (remove_chart_idx >= 0 && remove_chart_idx < static_cast<int>(tab->charts.size())) {
    tab->charts.erase(tab->charts.begin() + remove_chart_idx);
    state->cabana.active_chart_index = std::clamp(state->cabana.active_chart_index, 0,
                                                  std::max(0, static_cast<int>(tab->charts.size()) - 1));
  }
  if (drag_src_idx >= 0 && drag_dst_idx >= 0 && drag_src_idx != drag_dst_idx
      && drag_src_idx < static_cast<int>(tab->charts.size()) && drag_dst_idx < static_cast<int>(tab->charts.size())) {
    if (ImGui::GetIO().KeyShift) {
      CabanaChartState moved = std::move(tab->charts[static_cast<size_t>(drag_src_idx)]);
      tab->charts.erase(tab->charts.begin() + drag_src_idx);
      int dst = drag_insert_after ? drag_dst_idx + 1 : drag_dst_idx;
      if (drag_src_idx < dst) dst--;
      dst = std::clamp(dst, 0, static_cast<int>(tab->charts.size()));
      tab->charts.insert(tab->charts.begin() + dst, std::move(moved));
      state->cabana.active_chart_index = dst;
    } else {
      CabanaChartState &src = tab->charts[static_cast<size_t>(drag_src_idx)];
      CabanaChartState &dst = tab->charts[static_cast<size_t>(drag_dst_idx)];
      for (const std::string &path : src.signal_paths) {
        if (!chart_has_signal(dst, path)) {
          dst.signal_paths.push_back(path);
          dst.hidden.push_back(false);
        }
      }
      tab->charts.erase(tab->charts.begin() + drag_src_idx);
      state->cabana.active_chart_index = std::clamp(drag_dst_idx, 0, std::max(0, static_cast<int>(tab->charts.size()) - 1));
    }
  }

  ImGui::EndChild();
  ImGui::PopStyleColor();
  sync_chart_signal_aggregate(state);
}

namespace {

constexpr std::array<std::array<uint8_t, 3>, 8> kCabanaSignalPalette = {{
  {102, 86, 169},
  {69, 137, 255},
  {55, 171, 112},
  {232, 171, 44},
  {198, 89, 71},
  {92, 155, 181},
  {134, 172, 79},
  {150, 112, 63},
}};

bool signal_matches_filter(const CabanaSignalSummary &signal, std::string_view filter) {
  if (filter.empty()) {
    return true;
  }
  const std::string needle = lowercase(filter);
  return lowercase(signal.name).find(needle) != std::string::npos
      || lowercase(signal.unit).find(needle) != std::string::npos
      || lowercase(signal.receiver_name).find(needle) != std::string::npos;
}

bool cabana_signal_charted(const UiState &state, std::string_view path) {
  return std::find(state.cabana.chart_signal_paths.begin(), state.cabana.chart_signal_paths.end(), path)
      != state.cabana.chart_signal_paths.end();
}

const CabanaSignalSummary *find_message_signal(const CabanaMessageSummary &message, std::string_view path) {
  auto it = std::find_if(message.signals.begin(), message.signals.end(), [&](const CabanaSignalSummary &signal) {
    return signal.path == path;
  });
  return it == message.signals.end() ? nullptr : &*it;
}

void toggle_cabana_signal_chart(UiState *state, std::string_view path, bool enabled, bool new_chart_on_enable = false) {
  if (enabled && new_chart_on_enable) {
    ensure_chart_tabs(state);
    CabanaChartTabState *tab = active_chart_tab(state);
    if (tab != nullptr) {
      CabanaChartState chart{.id = state->cabana.next_chart_id++};
      chart.signal_paths.push_back(std::string(path));
      chart.hidden.push_back(false);
      tab->charts.push_back(std::move(chart));
      state->cabana.active_chart_index = static_cast<int>(tab->charts.size()) - 1;
      sync_chart_signal_aggregate(state);
    }
    return;
  }
  auto &paths = state->cabana.chart_signal_paths;
  auto it = std::find(paths.begin(), paths.end(), path);
  if (enabled) {
    if (it == paths.end()) {
      paths.emplace_back(path);
    }
  } else if (it != paths.end()) {
    paths.erase(it);
  }
}

void load_inline_signal_editor(UiState *state,
                               const CabanaMessageSummary &message,
                               const CabanaSignalSummary &signal) {
  CabanaSignalEditorState &editor = state->cabana_signal_editor;
  editor.open = false;
  editor.loaded = true;
  editor.creating = false;
  editor.message_root = message.root_path;
  editor.message_name = message.name;
  editor.service = message.service;
  editor.signal_path = signal.path;
  editor.bus = message.bus;
  editor.message_address = message.address;
  editor.original_signal_name = signal.name;
  editor.signal_name = signal.name;
  editor.start_bit = signal.start_bit;
  editor.size = signal.size;
  editor.factor = signal.factor;
  editor.offset = signal.offset;
  editor.min = signal.min;
  editor.max = signal.max;
  editor.is_signed = signal.is_signed;
  editor.is_little_endian = signal.is_little_endian;
  editor.type = signal.type;
  editor.multiplex_value = signal.multiplex_value;
  editor.receiver_name = signal.receiver_name;
  editor.unit = signal.unit;
}

void start_inline_signal_create(UiState *state,
                                const CabanaMessageSummary &message,
                                int start_bit,
                                int size,
                                bool is_little_endian) {
  const int byte_index = start_bit / 8;
  const int bit_index = start_bit & 7;
  std::string base_name = "bit_" + std::to_string(byte_index) + "_" + std::to_string(bit_index);
  std::string signal_name = base_name;
  int suffix = 2;
  auto exists = [&](std::string_view candidate) {
    return std::any_of(message.signals.begin(), message.signals.end(), [&](const CabanaSignalSummary &signal) {
      return signal.name == candidate;
    });
  };
  while (exists(signal_name)) {
    signal_name = base_name + "_" + std::to_string(suffix++);
  }

  CabanaSignalEditorState &editor = state->cabana_signal_editor;
  editor.open = false;
  editor.loaded = true;
  editor.creating = true;
  editor.message_root = message.root_path;
  editor.message_name = message.name;
  editor.service = message.service;
  editor.signal_path.clear();
  editor.bus = message.bus;
  editor.message_address = message.address;
  editor.original_signal_name.clear();
  editor.signal_name = signal_name;
  editor.start_bit = start_bit;
  editor.size = size;
  editor.factor = 1.0;
  editor.offset = 0.0;
  editor.min = 0.0;
  editor.max = std::min(std::pow(2.0, static_cast<double>(std::min(size, 24))) - 1.0, 1.0e9);
  editor.is_signed = false;
  editor.is_little_endian = is_little_endian;
  editor.type = 0;
  editor.multiplex_value = 0;
  editor.receiver_name = "XXX";
  editor.unit.clear();
}

const char *signal_type_label(int type) {
  switch (static_cast<dbc::Signal::Type>(type)) {
    case dbc::Signal::Type::Normal: return "Normal";
    case dbc::Signal::Type::Multiplexed: return "Muxed";
    case dbc::Signal::Type::Multiplexor: return "Mux";
  }
  return "?";
}

void draw_signal_list_header(UiState *state, const CabanaMessageSummary &message) {
  const size_t charted = state->cabana.chart_signal_paths.size();
  ImGui::PushStyleColor(ImGuiCol_ChildBg, cabana_panel_alt_bg());
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(6.0f, 4.0f));
  ImGui::BeginChild("##cabana_signals_header", ImVec2(0.0f, 34.0f), false, ImGuiWindowFlags_NoScrollbar);
  if (ImGui::BeginTable("##cabana_signal_header_layout", 3,
                        ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_NoPadOuterX | ImGuiTableFlags_NoPadInnerX)) {
    ImGui::TableSetupColumn("##left", ImGuiTableColumnFlags_WidthFixed, 132.0f);
    ImGui::TableSetupColumn("##filter", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn("##right", ImGuiTableColumnFlags_WidthFixed, 182.0f);
    ImGui::TableNextRow();

    ImGui::TableSetColumnIndex(0);
    app_push_bold_font();
    ImGui::Text("Signals: %zu", message.signals.size());
    app_pop_bold_font();
    if (charted > 0) {
      ImGui::SameLine(0.0f, 8.0f);
      ImGui::TextDisabled("%zu charted", charted);
    }

    ImGui::TableSetColumnIndex(1);
    const bool show_clear = state->cabana.signal_filter[0] != '\0';
    ImGui::SetNextItemWidth(show_clear ? -24.0f : -FLT_MIN);
    ImGui::InputTextWithHint("##cabana_signal_filter", "Filter signal / unit / receiver", state->cabana.signal_filter.data(),
                             state->cabana.signal_filter.size());
    if (show_clear) {
      ImGui::SameLine(0.0f, 4.0f);
      if (ImGui::SmallButton("x##clear_signal_filter")) {
        state->cabana.signal_filter[0] = '\0';
      }
    }

    ImGui::TableSetColumnIndex(2);
    ImGui::TextDisabled("%ds", state->cabana.sparkline_range_sec);
    ImGui::SameLine(0.0f, 4.0f);
    ImGui::SetNextItemWidth(76.0f);
    ImGui::SliderInt("##cabana_sparkline_range", &state->cabana.sparkline_range_sec, 1, 30, "");
    ImGui::SameLine(0.0f, 6.0f);
    if (ImGui::SmallButton("Collapse")) {
      state->cabana.selected_signal_path.clear();
      state->cabana_signal_editor.loaded = false;
      state->cabana_signal_editor.creating = false;
    }
    ImGui::EndTable();
  }
  ImGui::EndChild();
  ImGui::PopStyleVar();
  ImGui::PopStyleColor();
  ImGui::Spacing();
}

void draw_signal_inspector(AppSession *session,
                           UiState *state,
                           const CabanaMessageSummary &message,
                           const CabanaSignalSummary *selected_signal,
                           bool inline_mode = false) {
  CabanaSignalEditorState &editor = state->cabana_signal_editor;
  const bool showing_create = editor.loaded && editor.creating && editor.message_root == message.root_path;
  if (selected_signal == nullptr && !showing_create) {
    if (!inline_mode) {
      draw_empty_panel("Signal Inspector", "Select a decoded signal to inspect and edit it.");
    }
    return;
  }
  if (selected_signal != nullptr
      && (!editor.loaded || editor.creating || editor.message_root != message.root_path || editor.signal_path != selected_signal->path)) {
    load_inline_signal_editor(state, message, *selected_signal);
  }

  const bool charted = !editor.creating && !editor.signal_path.empty() && cabana_signal_charted(*state, editor.signal_path);
  if (!inline_mode) {
    draw_cabana_panel_title(showing_create ? "New Signal" : "Signal Inspector",
                            showing_create ? "Create and save directly into the active DBC" : std::string_view{});
  }
  ImGui::PushStyleColor(ImGuiCol_ChildBg, cabana_panel_alt_bg());
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8.0f, 6.0f));
  ImGui::BeginChild(inline_mode ? "##cabana_signal_editor_inline" : "##cabana_signal_editor", ImVec2(0.0f, 0.0f), true);

  app_push_bold_font();
  ImGui::TextUnformatted(showing_create ? "New Signal" : editor.signal_name.c_str());
  app_pop_bold_font();
  if (!editor.unit.empty()) {
    ImGui::SameLine(0.0f, 8.0f);
    ImGui::TextDisabled("[%s]", editor.unit.c_str());
  }
  if (!showing_create && selected_signal != nullptr) {
    ImGui::SameLine(0.0f, 10.0f);
    ImGui::TextDisabled("%s", signal_type_label(selected_signal->type));
    ImGui::SameLine(0.0f, 10.0f);
    const auto value = cabana_chart_value_label(*session, selected_signal->path, state->tracker_time);
    ImGui::TextDisabled("value %s", value.c_str());
  }
  if (!showing_create && !editor.signal_path.empty()) {
    ImGui::SameLine(0.0f, 12.0f);
    bool plot = charted;
    if (ImGui::Checkbox("Plot", &plot)) {
      toggle_cabana_signal_chart(state, editor.signal_path, plot, plot && !charted);
    }
  }

  if (selected_signal != nullptr && !selected_signal->comment.empty()) {
    ImGui::TextWrapped("%s", selected_signal->comment.c_str());
  } else if (selected_signal != nullptr && selected_signal->value_description_count > 0) {
    ImGui::TextDisabled("%d value description%s", selected_signal->value_description_count,
                        selected_signal->value_description_count == 1 ? "" : "s");
  }
  ImGui::Spacing();

  if (ImGui::BeginTable("##cabana_signal_editor_form", 2,
                        ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV)) {
    ImGui::TableSetupColumn("##left", ImGuiTableColumnFlags_WidthStretch, 1.0f);
    ImGui::TableSetupColumn("##right", ImGuiTableColumnFlags_WidthStretch, 1.0f);
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    input_text_string("Name", &editor.signal_name, ImGuiInputTextFlags_AutoSelectAll);
    ImGui::SetNextItemWidth(-FLT_MIN);
    ImGui::InputInt("Start Bit", &editor.start_bit);
    ImGui::SetNextItemWidth(-FLT_MIN);
    ImGui::InputInt("Size", &editor.size);
    ImGui::Checkbox("Little Endian", &editor.is_little_endian);
    ImGui::Checkbox("Signed", &editor.is_signed);

    ImGui::TableNextColumn();
    ImGui::SetNextItemWidth(-FLT_MIN);
    ImGui::InputDouble("Factor", &editor.factor, 0.0, 0.0, "%.6g");
    ImGui::SetNextItemWidth(-FLT_MIN);
    ImGui::InputDouble("Offset", &editor.offset, 0.0, 0.0, "%.6g");
    ImGui::SetNextItemWidth(-FLT_MIN);
    ImGui::InputDouble("Min", &editor.min, 0.0, 0.0, "%.6g");
    ImGui::SetNextItemWidth(-FLT_MIN);
    ImGui::InputDouble("Max", &editor.max, 0.0, 0.0, "%.6g");
    input_text_string("Unit", &editor.unit);
    input_text_string("Receiver", &editor.receiver_name);

    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    static constexpr const char *kTypes[] = {"Normal", "Multiplexed", "Multiplexor"};
    ImGui::SetNextItemWidth(-FLT_MIN);
    ImGui::Combo("Type", &editor.type, kTypes, IM_ARRAYSIZE(kTypes));
    if (editor.type != 0) {
      ImGui::SetNextItemWidth(-FLT_MIN);
      ImGui::InputInt("Mux Value", &editor.multiplex_value);
    }
    ImGui::EndTable();
  }

  ImGui::Spacing();
  if (ImGui::Button("Apply", ImVec2(96.0f, 0.0f))) {
    state->cabana.pending_apply_signal_edit = true;
  }
  ImGui::SameLine();
  if (ImGui::Button("Revert", ImVec2(96.0f, 0.0f))) {
    if (selected_signal != nullptr) {
      load_inline_signal_editor(state, message, *selected_signal);
    } else if (showing_create) {
      const int start_bit = state->cabana.has_bit_selection
        ? state->cabana.selected_bit_byte * 8 + state->cabana.selected_bit_index
        : 0;
      start_inline_signal_create(state, message, start_bit, 1, true);
    }
  }
  ImGui::SameLine();
  if (ImGui::Button("Raw DBC...", ImVec2(110.0f, 0.0f))) {
    state->dbc_editor.open = true;
    state->dbc_editor.loaded = false;
  }
  if (showing_create) {
    ImGui::SameLine();
    if (ImGui::Button("Cancel", ImVec2(96.0f, 0.0f))) {
      editor.loaded = false;
      editor.creating = false;
    }
  }

  ImGui::EndChild();
  ImGui::PopStyleVar();
  ImGui::PopStyleColor();
}

}  // namespace

void draw_signal_panel(AppSession *session, UiState *state, const CabanaMessageSummary &message) {
  draw_signal_list_header(state, message);
  if (message.signals.empty() && !(state->cabana_signal_editor.loaded && state->cabana_signal_editor.creating)) {
    draw_empty_panel("Signals", "No decoded signals for this message.");
    return;
  }

  const std::string filter = trim_copy(state->cabana.signal_filter.data());
  std::vector<size_t> visible_indices;
  visible_indices.reserve(message.signals.size());
  for (size_t i = 0; i < message.signals.size(); ++i) {
    if (signal_matches_filter(message.signals[i], filter)) {
      visible_indices.push_back(i);
    }
  }

  const CabanaSignalSummary *selected_signal = find_message_signal(message, state->cabana.selected_signal_path);
  const bool show_create_editor = state->cabana_signal_editor.loaded
                               && state->cabana_signal_editor.creating
                               && state->cabana_signal_editor.message_root == message.root_path;

  ImGui::BeginChild("##cabana_signal_list", ImVec2(0.0f, 0.0f), false);
  if (show_create_editor && selected_signal == nullptr) {
    ImGui::BeginChild("##cabana_signal_create_inline", ImVec2(0.0f, 228.0f), false);
    draw_signal_inspector(session, state, message, nullptr, true);
    ImGui::EndChild();
    ImGui::Dummy(ImVec2(0.0f, 6.0f));
  }
  if (visible_indices.empty()) {
    ImGui::TextDisabled("No signals match this filter.");
  } else {
    for (size_t row = 0; row < visible_indices.size(); ++row) {
        const size_t visible_index = visible_indices[row];
        const CabanaSignalSummary &signal = message.signals[visible_index];
        const bool charted = cabana_signal_charted(*state, signal.path);
        const bool selected = state->cabana.selected_signal_path == signal.path;

        ImGui::PushID(signal.path.c_str());
        const ImVec2 row_pos = ImGui::GetCursorScreenPos();
        const float row_w = ImGui::GetContentRegionAvail().x;
        const float row_h = 28.0f;
        ImGui::Dummy(ImVec2(row_w, row_h));
        const ImRect row_rect(row_pos, ImVec2(row_pos.x + row_w, row_pos.y + row_h));
        const bool row_hovered = ImGui::IsMouseHoveringRect(row_rect.Min, row_rect.Max);
        ImDrawList *draw = ImGui::GetWindowDrawList();
        const ImU32 row_bg = ImGui::GetColorU32(selected ? color_rgb(66, 84, 117) : (row_hovered ? color_rgb(66, 68, 72) : color_rgb(58, 61, 64)));
        const ImU32 row_border = ImGui::GetColorU32(selected ? color_rgb(108, 145, 214) : color_rgb(86, 90, 96));
        draw->AddRectFilled(row_rect.Min, row_rect.Max, row_bg, 3.0f);
        draw->AddRect(row_rect.Min, row_rect.Max, row_border, 3.0f);

        const float button_w = 22.0f;
        const float value_w = 82.0f;
        const float spark_w = std::clamp(row_w * 0.28f, 110.0f, 180.0f);
        const ImRect delete_rect(ImVec2(row_rect.Max.x - 6.0f - button_w, row_rect.Min.y + 3.0f),
                                 ImVec2(row_rect.Max.x - 6.0f, row_rect.Max.y - 3.0f));
        const ImRect edit_rect(ImVec2(delete_rect.Min.x - 4.0f - button_w, row_rect.Min.y + 3.0f),
                               ImVec2(delete_rect.Min.x - 4.0f, row_rect.Max.y - 3.0f));
        const ImRect plot_rect(ImVec2(edit_rect.Min.x - 4.0f - button_w, row_rect.Min.y + 3.0f),
                               ImVec2(edit_rect.Min.x - 4.0f, row_rect.Max.y - 3.0f));
        const ImRect value_rect(ImVec2(plot_rect.Min.x - 8.0f - value_w, row_rect.Min.y),
                                ImVec2(plot_rect.Min.x - 8.0f, row_rect.Max.y));
        const ImRect spark_rect(ImVec2(std::max(row_rect.Min.x + 180.0f, value_rect.Min.x - 8.0f - spark_w), row_rect.Min.y + 2.0f),
                                ImVec2(value_rect.Min.x - 8.0f, row_rect.Max.y - 2.0f));

        const float badge_x = row_rect.Min.x + 8.0f;
        const ImRect badge_rect(ImVec2(badge_x, row_rect.Min.y + 4.0f), ImVec2(badge_x + 22.0f, row_rect.Max.y - 4.0f));
        const ImU32 badge_fill = ImGui::GetColorU32(color_rgb(kCabanaSignalPalette[visible_index % kCabanaSignalPalette.size()]));
        draw->AddRectFilled(badge_rect.Min, badge_rect.Max, badge_fill, 3.0f);
        const std::string ordinal = std::to_string(static_cast<int>(row) + 1);
        const ImVec2 ordinal_size = ImGui::CalcTextSize(ordinal.c_str());
        draw->AddText(ImVec2(badge_rect.Min.x + (badge_rect.GetWidth() - ordinal_size.x) * 0.5f,
                             badge_rect.Min.y + (badge_rect.GetHeight() - ordinal_size.y) * 0.5f - 1.0f),
                      ImGui::GetColorU32(IM_COL32_BLACK), ordinal.c_str());

        float text_x = badge_rect.Max.x + 8.0f;
        if (signal.type != 0) {
          const std::string mux = signal.type == static_cast<int>(dbc::Signal::Type::Multiplexor)
                                ? "M"
                                : ("m" + std::to_string(signal.multiplex_value));
          const ImVec2 mux_size = ImGui::CalcTextSize(mux.c_str());
          const ImRect mux_rect(ImVec2(text_x, row_rect.Min.y + 5.0f),
                                ImVec2(text_x + mux_size.x + 10.0f, row_rect.Max.y - 5.0f));
          draw->AddRectFilled(mux_rect.Min, mux_rect.Max, ImGui::GetColorU32(color_rgb(118, 122, 128)), 3.0f);
          draw->AddText(ImVec2(mux_rect.Min.x + 5.0f, mux_rect.Min.y + (mux_rect.GetHeight() - mux_size.y) * 0.5f - 1.0f),
                        ImGui::GetColorU32(color_rgb(238, 240, 242)),
                        mux.c_str());
          text_x = mux_rect.Max.x + 8.0f;
        }

        const ImRect name_rect(ImVec2(text_x, row_rect.Min.y), ImVec2(std::max(text_x, spark_rect.Min.x - 10.0f), row_rect.Max.y));
        const std::string name = signal.name;
        const ImVec2 name_size = ImGui::CalcTextSize(name.c_str());
        const float name_max_w = std::max(0.0f, name_rect.GetWidth());
        std::string name_text = name;
        if (name_size.x > name_max_w) {
          name_text = name.substr(0, std::min(name.size(), static_cast<size_t>(std::max(1.0f, name_max_w / 7.0f)))) + "...";
        }
        draw->AddText(ImVec2(name_rect.Min.x, row_rect.Min.y + 6.0f),
                      ImGui::GetColorU32(color_rgb(225, 229, 233)),
                      name_text.c_str());

        ImGui::SetCursorScreenPos(spark_rect.Min);
        draw_signal_sparkline(*session, *state, signal.path, selected || charted, spark_rect.GetSize());
        draw->AddText(ImVec2(value_rect.Min.x, row_rect.Min.y + 6.0f),
                      ImGui::GetColorU32(color_rgb(207, 212, 218)),
                      cabana_chart_value_label(*session, signal.path, state->tracker_time).c_str());

        auto draw_row_button = [&](const char *id, const ImRect &rect, const char *glyph, bool active, const char *tooltip) {
          ImGui::SetCursorScreenPos(rect.Min);
          ImGui::InvisibleButton(id, rect.GetSize());
          const bool hovered = ImGui::IsItemHovered();
          draw->AddRectFilled(rect.Min, rect.Max,
                              ImGui::GetColorU32(active ? cabana_accent() : (hovered ? color_rgb(82, 86, 91) : color_rgb(67, 70, 74))),
                              3.0f);
          draw->AddRect(rect.Min, rect.Max, ImGui::GetColorU32(color_rgb(96, 101, 108)), 3.0f);
          ImVec2 text_size = ImGui::CalcTextSize(glyph);
          draw->AddText(ImVec2(rect.Min.x + (rect.GetWidth() - text_size.x) * 0.5f,
                               rect.Min.y + (rect.GetHeight() - text_size.y) * 0.5f - 1.0f),
                        ImGui::GetColorU32(color_rgb(236, 239, 242)),
                        glyph);
          if (hovered && tooltip != nullptr) {
            ImGui::SetTooltip("%s", tooltip);
          }
          return ImGui::IsItemClicked(ImGuiMouseButton_Left);
        };

        const bool plot_clicked = draw_row_button("##plot", plot_rect, icon::BAR_CHART, charted,
                                                  charted ? "Close Plot" : "Show Plot");
        const bool edit_clicked = draw_row_button("##edit", edit_rect, icon::SLIDERS, false, "Inspect / Edit Signal");
        const bool delete_clicked = draw_row_button("##delete", delete_rect, "x", false, "Delete Signal");

        if (plot_clicked) {
          toggle_cabana_signal_chart(state, signal.path, !charted, !charted);
        }
        if (edit_clicked) {
          if (selected) {
            state->cabana.selected_signal_path.clear();
            state->cabana_signal_editor.loaded = false;
          } else {
            state->cabana.selected_signal_path = signal.path;
          }
        }
        if (delete_clicked) {
          load_inline_signal_editor(state, message, signal);
          state->cabana.pending_delete_signal = true;
        }

        const bool row_clickable = row_hovered
                                && !ImGui::IsMouseHoveringRect(plot_rect.Min, plot_rect.Max)
                                && !ImGui::IsMouseHoveringRect(edit_rect.Min, edit_rect.Max)
                                && !ImGui::IsMouseHoveringRect(delete_rect.Min, delete_rect.Max);
        if (row_clickable && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
          if (selected) {
            state->cabana.selected_signal_path.clear();
            state->cabana_signal_editor.loaded = false;
          } else {
            state->cabana.selected_signal_path = signal.path;
          }
        }
        if (row_clickable && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
          toggle_cabana_signal_chart(state, signal.path, !charted, !charted);
        }
        if (selected) {
          ImGui::Dummy(ImVec2(0.0f, 4.0f));
          ImGui::BeginChild("##cabana_signal_inline_editor", ImVec2(0.0f, 228.0f), false);
          draw_signal_inspector(session, state, message, &signal, true);
          ImGui::EndChild();
          ImGui::Dummy(ImVec2(0.0f, 6.0f));
        }
        ImGui::PopID();
    }
  }
  ImGui::EndChild();
}
