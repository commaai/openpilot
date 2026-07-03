#include "tools/cabana/imgui/app.h"

void draw_messages_panel(AppState & /*app*/) {
  if (ImGui::Begin(MESSAGES_WINDOW_TITLE)) {
    ImGui::TextDisabled("Messages table -- coming soon");
  }
  ImGui::End();
}
