#include "tools/cabana/imgui/app.h"

void draw_charts_panel(AppState & /*app*/) {
  if (ImGui::Begin(CHARTS_WINDOW_TITLE)) {
    ImGui::TextDisabled("Charts -- coming soon");
  }
  ImGui::End();
}
