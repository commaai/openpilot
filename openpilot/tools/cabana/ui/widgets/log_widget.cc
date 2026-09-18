#include "tools/cabana/ui/widgets/log_widget.h"

#include <algorithm>
#include <cstdio>

#include "imgui.h"
#include "tools/cabana/streams/abstractstream.h"

namespace cabana {

LogWidget::LogWidget() = default;

void LogWidget::updateLogs() {
  // Logs are updated when new segments arrive
  const auto snapshot = can->cereal_series.snapshot();
  if (snapshot.revision == last_revision_) return;
  last_revision_ = snapshot.revision;
}

void LogWidget::draw() {
  if (!visible) return;

  ImGui::SetNextWindowSize(ImVec2(500, 300), ImGuiCond_FirstUseEver);
  if (!ImGui::Begin("Logs###LogsWidget", &visible)) {
    ImGui::End();
    return;
  }

  updateLogs();

  filter_.Draw("Filter", 180.0f);
  ImGui::SameLine();
  ImGui::Checkbox("Debug", &show_debug_);
  ImGui::SameLine();
  ImGui::Checkbox("Info", &show_info_);
  ImGui::SameLine();
  ImGui::Checkbox("Warn", &show_warning_);
  ImGui::SameLine();
  ImGui::Checkbox("Error", &show_error_);

  ImGui::Separator();

  static const ImGuiTableFlags table_flags =
      ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersInnerH |
      ImGuiTableFlags_Resizable | ImGuiTableFlags_ScrollY;

  if (ImGui::BeginTable("##logs_table", 4, table_flags)) {
    ImGui::TableSetupColumn("Time", ImGuiTableColumnFlags_WidthFixed, 60.0f);
    ImGui::TableSetupColumn("Level", ImGuiTableColumnFlags_WidthFixed, 50.0f);
    ImGui::TableSetupColumn("Source", ImGuiTableColumnFlags_WidthFixed, 100.0f);
    ImGui::TableSetupColumn("Message", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableHeadersRow();

    const double cur_sec = can->currentSec();
    for (size_t i = 0; i < logs_.size(); ++i) {
      const auto &item = logs_[i];
      if (item.level < 20 && !show_debug_) continue;
      if (item.level >= 20 && item.level < 30 && !show_info_) continue;
      if (item.level >= 30 && item.level < 40 && !show_warning_) continue;
      if (item.level >= 40 && !show_error_) continue;
      if (!filter_.PassFilter(item.message.c_str()) && !filter_.PassFilter(item.source.c_str())) continue;

      ImGui::TableNextRow();
      const bool is_near_cursor = std::abs(item.seconds - cur_sec) < 0.5;
      if (is_near_cursor) {
        ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, ImGui::GetColorU32(ImGuiCol_Header));
      }

      ImGui::TableNextColumn();
      char time_buf[32];
      snprintf(time_buf, sizeof(time_buf), "%.2f", item.seconds);
      if (ImGui::Selectable(time_buf, false, ImGuiSelectableFlags_SpanAllColumns)) {
        can->seekTo(item.seconds);
      }

      ImGui::TableNextColumn();
      const char *level_str = item.level >= 40 ? "ERROR" : item.level >= 30 ? "WARN" : item.level >= 20 ? "INFO" : "DEBUG";
      ImGui::TextUnformatted(level_str);

      ImGui::TableNextColumn();
      ImGui::TextUnformatted(item.source.c_str());

      ImGui::TableNextColumn();
      ImGui::TextUnformatted(item.message.c_str());
    }

    ImGui::EndTable();
  }

  ImGui::End();
}

}  // namespace cabana
