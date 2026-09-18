#include "tools/jotpluggler/camera.h"

#include "imgui.h"
#include "imgui_internal.h"

namespace {

bool draw_camera_fit_toggle_overlay(bool fit_to_pane) {
  const ImVec2 window_pos = ImGui::GetWindowPos();
  const ImVec2 content_min = ImGui::GetWindowContentRegionMin();
  const ImRect rect(ImVec2(window_pos.x + content_min.x + 8.0f, window_pos.y + content_min.y + 8.0f),
                    ImVec2(window_pos.x + content_min.x + 58.0f, window_pos.y + content_min.y + 28.0f));
  const bool hovered = ImGui::IsMouseHoveringRect(rect.Min, rect.Max, false);
  const bool held = hovered && ImGui::IsMouseDown(ImGuiMouseButton_Left);
  if (hovered) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);

  ImDrawList *draw_list = ImGui::GetWindowDrawList();
  draw_list->AddRectFilled(rect.Min, rect.Max, hovered ? IM_COL32(255, 255, 255, 234) : IM_COL32(255, 255, 255, 214), 4.0f);
  draw_list->AddRect(rect.Min, rect.Max, IM_COL32(184, 189, 196, 255), 4.0f, 0, 1.0f);
  const ImRect box(ImVec2(rect.Min.x + 6.0f, rect.Min.y + 4.0f), ImVec2(rect.Min.x + 18.0f, rect.Min.y + 16.0f));
  draw_list->AddRect(box.Min, box.Max, IM_COL32(112, 120, 129, 255), 2.0f, 0, 1.0f);
  if (fit_to_pane) {
    draw_list->AddLine(ImVec2(box.Min.x + 2.5f, box.Min.y + 6.5f), ImVec2(box.Min.x + 5.5f, box.Max.y - 2.5f), IM_COL32(60, 111, 202, 255), 1.8f);
    draw_list->AddLine(ImVec2(box.Min.x + 5.5f, box.Max.y - 2.5f), ImVec2(box.Max.x - 2.5f, box.Min.y + 2.5f), IM_COL32(60, 111, 202, 255), 1.8f);
  }
  draw_list->AddText(ImVec2(box.Max.x + 6.0f, rect.Min.y + 3.0f), IM_COL32(72, 79, 88, 255), "Fit");
  return hovered && !held && ImGui::IsMouseReleased(ImGuiMouseButton_Left);
}

}  // namespace

void draw_camera_pane(AppSession *session, UiState *state, TabUiState *tab_state, int pane_index, const Pane &pane) {
  CameraFeedView *feed = session->pane_camera_feeds[static_cast<size_t>(pane.camera_view)].get();
  if (feed == nullptr) {
    ImGui::TextDisabled("Camera unavailable");
    return;
  }

  const bool fit_to_pane = tab_state != nullptr
    && pane_index >= 0
    && pane_index < static_cast<int>(tab_state->camera_panes.size())
    ? tab_state->camera_panes[static_cast<size_t>(pane_index)].fit_to_pane
    : true;
  if (state->has_tracker_time) {
    feed->update(state->tracker_time);
  }
  feed->drawSized(ImGui::GetContentRegionAvail(), session->async_route_loading, fit_to_pane);
  if (tab_state != nullptr
      && pane_index >= 0
      && pane_index < static_cast<int>(tab_state->camera_panes.size())
      && draw_camera_fit_toggle_overlay(fit_to_pane)) {
    tab_state->camera_panes[static_cast<size_t>(pane_index)].fit_to_pane = !fit_to_pane;
  }
}
