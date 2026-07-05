#include "tools/loggy/panes/browser.h"

#include "tools/loggy/backend/session.h"
#include "tools/loggy/shell/theme.h"
#include "tools/loggy/shell/workspace.h"

#include "imgui.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <vector>

namespace loggy {
namespace {

void draw_browser_sparkline(const BrowserSparkline &sparkline) {
  constexpr float width = 92.0f;
  const float height = std::max(18.0f, ImGui::GetTextLineHeight() + 4.0f);
  const ImVec2 pos = ImGui::GetCursorScreenPos();
  ImGui::Dummy(ImVec2(width, height));

  ImDrawList *draw_list = ImGui::GetWindowDrawList();
  const ImVec2 max(pos.x + width, pos.y + height);
  draw_list->AddRectFilled(pos, max, ImGui::GetColorU32(color_rgb(48, 51, 53)), 2.0f);
  draw_list->AddRect(pos, max, ImGui::GetColorU32(color_rgb(82, 86, 88)), 2.0f);
  if (sparkline.values.empty()) {
    draw_list->AddText(ImVec2(pos.x + 4.0f, pos.y + 2.0f), ImGui::GetColorU32(ImGuiCol_TextDisabled), "--");
    return;
  }

  const double raw_span = sparkline.max - sparkline.min;
  const double span = std::max(raw_span, 1e-9);
  std::vector<ImVec2> points;
  points.reserve(sparkline.values.size());
  for (size_t i = 0; i < sparkline.values.size(); ++i) {
    const float x = pos.x + 2.0f + (width - 4.0f) * (sparkline.values.size() == 1 ? 0.5f : static_cast<float>(i) / static_cast<float>(sparkline.values.size() - 1));
    const double normalized = raw_span <= 1e-9 ? 0.5 : (sparkline.max - sparkline.values[i]) / span;
    const float y = pos.y + 2.0f + (height - 4.0f) * static_cast<float>(normalized);
    points.push_back(ImVec2(x, y));
  }
  if (points.size() == 1) {
    draw_list->AddCircleFilled(points.front(), 2.0f, ImGui::GetColorU32(color_rgb(116, 178, 255)));
  } else {
    draw_list->AddPolyline(points.data(), static_cast<int>(points.size()), ImGui::GetColorU32(color_rgb(116, 178, 255)), 0, 1.5f);
  }
}

void draw_browser_row(const BrowserSeriesRow &row) {
  ImGui::TableNextRow(ImGuiTableRowFlags_None, std::max(ImGui::GetFrameHeight(), ImGui::GetTextLineHeight() + 8.0f));
  ImGui::PushID(row.path.c_str());

  ImGui::TableSetColumnIndex(0);
  const bool selected = false;
  ImGui::Selectable(row.label.c_str(), selected, ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowOverlap);
  if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID)) {
    ImGui::SetDragDropPayload(kLoggySeriesPathPayload, row.path.c_str(), row.path.size() + 1);
    ImGui::TextUnformatted(row.path.c_str());
    ImGui::EndDragDropSource();
  }

  ImGui::TableSetColumnIndex(1);
  push_mono_font();
  ImGui::TextUnformatted(row.value.c_str());
  pop_mono_font();

  ImGui::TableSetColumnIndex(2);
  draw_browser_sparkline(row.sparkline);

  ImGui::TableSetColumnIndex(3);
  push_mono_font();
  ImGui::TextUnformatted(row.path.c_str());
  pop_mono_font();

  ImGui::PopID();
}

}  // namespace

void draw_browser_pane(Session &session, PaneInstance &pane) {
  BrowserState state = parse_browser_state(pane.state_json);
  bool changed = false;

  std::array<char, 160> filter_buf{};
  std::snprintf(filter_buf.data(), filter_buf.size(), "%s", state.filter.c_str());
  const float filter_width = std::clamp(ImGui::GetContentRegionAvail().x * 0.55f, 140.0f, 320.0f);
  ImGui::SetNextItemWidth(filter_width);
  if (ImGui::InputTextWithHint("Search", "Path", filter_buf.data(), filter_buf.size())) {
    state.filter = filter_buf.data();
    changed = true;
  }
  if (ImGui::GetContentRegionAvail().x > 170.0f) ImGui::SameLine();
  int sparkline_seconds = state.sparkline_seconds;
  ImGui::SetNextItemWidth(96.0f);
  if (ImGui::SliderInt("Spark", &sparkline_seconds, 1, 120, "%ds", ImGuiSliderFlags_AlwaysClamp)) {
    state.sparkline_seconds = sparkline_seconds;
    changed = true;
  }
  if (changed) pane.state_json = browser_state_json(state);

  const std::vector<BrowserSeriesRow> rows = prepare_browser_series_rows(session.store(), state);
  const size_t total = session.store().seriesPathCount();
  if (ImGui::GetContentRegionAvail().x > 120.0f) ImGui::SameLine();
  ImGui::TextDisabled("%zu/%zu series", rows.size(), total);

  if (rows.empty()) {
    ImGui::TextDisabled("No series in store or filter");
    return;
  }

  constexpr ImGuiTableFlags flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV |
                                    ImGuiTableFlags_BordersOuter | ImGuiTableFlags_SizingStretchProp |
                                    ImGuiTableFlags_ScrollY;
  if (!ImGui::BeginTable("##loggy_browser_series", 4, flags, ImGui::GetContentRegionAvail())) return;
  ImGui::TableSetupColumn("Field", ImGuiTableColumnFlags_WidthFixed, 126.0f);
  ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthFixed, 88.0f);
  ImGui::TableSetupColumn("Spark", ImGuiTableColumnFlags_WidthFixed, 102.0f);
  ImGui::TableSetupColumn("Path", ImGuiTableColumnFlags_WidthStretch);
  ImGui::TableHeadersRow();

  const TimeRange range = session.view_range().range();
  const double tracker_time = session.playback().tracker_time();
  ImGuiListClipper clipper;
  clipper.Begin(static_cast<int>(rows.size()), std::max(ImGui::GetFrameHeight(), ImGui::GetTextLineHeight() + 8.0f));
  while (clipper.Step()) {
    for (int row_idx = clipper.DisplayStart; row_idx < clipper.DisplayEnd; ++row_idx) {
      draw_browser_row(enrich_browser_series_row(session.store(), rows[static_cast<size_t>(row_idx)],
                                                 range, tracker_time, state));
    }
  }
  ImGui::EndTable();
}

}  // namespace loggy
