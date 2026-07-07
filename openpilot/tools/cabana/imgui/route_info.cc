// ImGui port of tools/cabana/tools/routeinfo.{h,cc} (RouteInfoDlg): a small
// modal listing each route segment's available file types, click a row to
// seek to that segment. Opened from transport.cc's "View route details"
// button (see that file for where Qt places the trigger).
#include "tools/cabana/imgui/app.h"

#include <algorithm>
#include <string>

#include "tools/cabana/streams/replaystream.h"

namespace {
bool g_want_open = false;
constexpr const char *kPopupId = "Route Info##route_info_dlg";
}  // namespace

void open_route_info() { g_want_open = true; }

void draw_route_info(AppState &app) {
  ReplayStream *rs = dynamic_cast<ReplayStream *>(app.stream.get());

  if (g_want_open) {
    g_want_open = false;
    if (rs != nullptr) ImGui::OpenPopup(kPopupId);
  }
  if (rs == nullptr) return;

  Replay *replay = rs->getReplay();
  ImGui::SetNextWindowSize(ImVec2(560.0f, 480.0f), ImGuiCond_Appearing);
  ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
  if (!ImGui::BeginPopupModal(kPopupId, nullptr, ImGuiWindowFlags_NoSavedSettings)) return;

  // mirrors RouteInfoDlg's window title ("Route: %1")
  ImGui::Text("Route: %s", replay->route().name().c_str());
  ImGui::TextDisabled("Click on a row to seek to the corresponding segment.");
  ImGui::Separator();

  const ImGuiTableFlags table_flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders |
                                       ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingFixedFit;
  const float row_h = ImGui::GetTextLineHeightWithSpacing();
  const float table_h = std::min(row_h * static_cast<float>(replay->route().segments().size() + 1), 420.0f);
  if (ImGui::BeginTable("##route_segments", 7, table_flags, ImVec2(0.0f, table_h))) {
    // exact column order/labels from RouteInfoDlg: {"", "rlog", "fcam", "ecam", "dcam", "qlog", "qcam"}
    ImGui::TableSetupColumn("#");
    ImGui::TableSetupColumn("rlog");
    ImGui::TableSetupColumn("fcam");
    ImGui::TableSetupColumn("ecam");
    ImGui::TableSetupColumn("dcam");
    ImGui::TableSetupColumn("qlog");
    ImGui::TableSetupColumn("qcam");
    ImGui::TableHeadersRow();

    const auto yes_no = [](const std::string &s) { return s.empty() ? "--" : "Yes"; };
    for (const auto &[seg_num, seg] : replay->route().segments()) {
      ImGui::TableNextRow();
      ImGui::TableSetColumnIndex(0);
      ImGui::PushID(seg_num);
      const std::string label = std::to_string(seg_num);
      if (ImGui::Selectable(label.c_str(), false, ImGuiSelectableFlags_SpanAllColumns)) {
        app.stream->seekTo(seg_num * 60.0);
      }
      ImGui::PopID();
      ImGui::TableSetColumnIndex(1);
      ImGui::TextUnformatted(yes_no(seg.rlog));
      ImGui::TableSetColumnIndex(2);
      ImGui::TextUnformatted(yes_no(seg.road_cam));
      ImGui::TableSetColumnIndex(3);
      ImGui::TextUnformatted(yes_no(seg.wide_road_cam));
      ImGui::TableSetColumnIndex(4);
      ImGui::TextUnformatted(yes_no(seg.driver_cam));
      ImGui::TableSetColumnIndex(5);
      ImGui::TextUnformatted(yes_no(seg.qlog));
      ImGui::TableSetColumnIndex(6);
      ImGui::TextUnformatted(yes_no(seg.qcamera));
    }
    ImGui::EndTable();
  }

  ImGui::Spacing();
  if (ImGui::Button("Close", ImVec2(100.0f, 0.0f)) || ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
    ImGui::CloseCurrentPopup();
  }
  ImGui::EndPopup();
}
