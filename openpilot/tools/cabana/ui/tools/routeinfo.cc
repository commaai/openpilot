#include "tools/cabana/ui/tools/routeinfo.h"

#include <algorithm>
#include <cstdio>

#include "imgui.h"
#include "tools/cabana/streams/replaystream.h"
#include "tools/cabana/ui/imgui_util.h"

RouteInfoDlg::RouteInfoDlg() {
  auto *replay = dynamic_cast<ReplayStream *>(can)->getReplay();
  char buf[64];
  snprintf(buf, sizeof(buf), "###routeinfo%p", (void *)this);
  title_ = "Route: " + replay->route().name() + buf;

  for (const auto &[seg_num, seg] : replay->route().segments()) {
    rows_.push_back({std::to_string(seg_num),
                     seg.rlog.empty() ? "--" : "Yes",
                     seg.narrow_road_cam.empty() ? "--" : "Yes",
                     seg.wide_road_cam.empty() ? "--" : "Yes",
                     seg.cabin_cam.empty() ? "--" : "Yes",
                     seg.qlog.empty() ? "--" : "Yes",
                     seg.qcamera.empty() ? "--" : "Yes"});
  }
}

bool RouteInfoDlg::draw() {
  if (!open_) return false;
  static const char *headers[] = {"", "rlog", "narrow road", "wide road", "driver", "qlog", "qcam"};
  // minimum size: header + min(rowCount, 13) rows
  float row_h = ImGui::GetTextLineHeightWithSpacing();
  float min_h = row_h * (std::min((int)rows_.size(), 13) + 1) + ImGui::GetFrameHeightWithSpacing() + ImGui::GetStyle().WindowPadding.y * 2;
  ImGui::SetNextWindowSize(ImVec2(520, min_h), ImGuiCond_Appearing);
  if (ImGui::Begin(title_.c_str(), &open_)) {
    const ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingFixedFit;
    if (ImGui::BeginTable("table", 7, flags, ImVec2(0, 0))) {
      ImGui::TableSetupScrollFreeze(0, 1);
      for (int c = 0; c < 7; ++c) ImGui::TableSetupColumn(headers[c]);
      tableHeadersRow();
      for (int row = 0; row < (int)rows_.size(); ++row) {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::PushID(row);
        if (ImGui::Selectable(rows_[row][0].c_str(), false, ImGuiSelectableFlags_SpanAllColumns)) {
          can->seekTo(row * 60.0);
        }
        ImGui::SetItemTooltip("Click on a row to seek to the corresponding segment.");
        ImGui::PopID();
        for (int c = 1; c < 7; ++c) {
          ImGui::TableSetColumnIndex(c);
          ImGui::TextUnformatted(rows_[row][c].c_str());
        }
      }
      ImGui::EndTable();
    }
  }
  ImGui::End();
  return open_;
}
