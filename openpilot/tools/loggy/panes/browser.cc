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
  if (!ImGui::BeginTable("##loggy_browser_series", 2, flags, ImGui::GetContentRegionAvail())) return;
  ImGui::TableSetupColumn("Field", ImGuiTableColumnFlags_WidthFixed, 126.0f);
  ImGui::TableSetupColumn("Path", ImGuiTableColumnFlags_WidthStretch);
  ImGui::TableHeadersRow();

  ImGuiListClipper clipper;
  clipper.Begin(static_cast<int>(rows.size()), std::max(ImGui::GetFrameHeight(), ImGui::GetTextLineHeight() + 8.0f));
  while (clipper.Step()) {
    for (int row_idx = clipper.DisplayStart; row_idx < clipper.DisplayEnd; ++row_idx) {
      draw_browser_row(rows[static_cast<size_t>(row_idx)]);
    }
  }
  ImGui::EndTable();
}

}  // namespace loggy
