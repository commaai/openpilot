#include "tools/cabana/imgui/app.h"

void draw_charts_panel(AppState & /*app*/) {
  if (ImGui::Begin(CHARTS_WINDOW_TITLE)) {
    ImGui::TextDisabled("Charts -- coming soon");
  }
  ImGui::End();
}

void charts_show_signal(const MessageId &id, const cabana::Signal *sig, bool show) {}

bool charts_is_showing(const MessageId &id, const cabana::Signal *sig) {
  return false;
}
