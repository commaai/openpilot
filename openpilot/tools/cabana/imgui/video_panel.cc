#include "tools/cabana/imgui/app.h"

void draw_video_panel(AppState & /*app*/) {
  if (ImGui::Begin(VIDEO_WINDOW_TITLE)) {
    ImGui::TextDisabled("Camera view -- coming soon");
  }
  ImGui::End();
}
