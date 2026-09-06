#include "tools/cabana/ui/tools/routeinfo.h"

#include <algorithm>
#include <string>

#include "imgui.h"
#include "tools/cabana/streams/replaystream.h"
#include "tools/cabana/ui/util.h"

RouteInfoDlg::RouteInfoDlg() {
  replay_ = dynamic_cast<ReplayStream *>(can)->getReplay();
  setTitle("Route: " + replay_->route().name());
}

bool RouteInfoDlg::draw() {
  static const char *headers[] = {"", "rlog", "narrow road", "wide road", "driver", "qlog", "qcam"};
  auto yn = [](const std::string &s) { return s.empty() ? "--" : "Yes"; };
  const auto &segments = replay_->route().segments();
  // minimum size: header + min(rowCount, 13) rows
  float row_h = ImGui::GetTextLineHeightWithSpacing();
  float min_h = row_h * (std::min((int)segments.size(), 13) + 1) + ImGui::GetFrameHeightWithSpacing() + ImGui::GetStyle().WindowPadding.y * 2;
  if (begin(ImVec2(520, min_h))) {
    const ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingFixedFit;
    if (ImGui::BeginTable("table", 7, flags, ImVec2(0, 0))) {
      ImGui::TableSetupScrollFreeze(0, 1);
      for (int c = 0; c < 7; ++c) ImGui::TableSetupColumn(headers[c]);
      tableHeadersRow();
      int row = 0;
      for (const auto &[seg_num, seg] : segments) {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::PushID(row);
        if (ImGui::Selectable(std::to_string(seg_num).c_str(), false, ImGuiSelectableFlags_SpanAllColumns)) {
          can->seekTo(row * 60.0);
        }
        ImGui::SetItemTooltip("Click on a row to seek to the corresponding segment.");
        ImGui::PopID();
        const char *cells[] = {yn(seg.rlog), yn(seg.narrow_road_cam), yn(seg.wide_road_cam),
                               yn(seg.cabin_cam), yn(seg.qlog), yn(seg.qcamera)};
        for (int c = 1; c < 7; ++c) {
          ImGui::TableSetColumnIndex(c);
          ImGui::TextUnformatted(cells[c - 1]);
        }
        ++row;
      }
      ImGui::EndTable();
    }
  }
  return end();
}
