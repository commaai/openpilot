#include "tools/cabana/imgui/app.h"

#include <algorithm>

namespace {

void draw_centered_text(const char *text) {
  const float width = ImGui::GetContentRegionAvail().x;
  ImGui::SetCursorPosX((width - ImGui::CalcTextSize(text).x) * 0.5f);
  ImGui::TextUnformatted(text);
}

void draw_shortcut_row(const char *title, const char *key) {
  const float center = ImGui::GetContentRegionAvail().x * 0.5f;
  const float title_w = ImGui::CalcTextSize(title).x;
  ImGui::SetCursorPosX(center - title_w - ImGui::GetStyle().ItemSpacing.x);
  ImGui::AlignTextToFramePadding();
  ImGui::TextDisabled("%s", title);
  ImGui::SameLine(center);
  ImGui::BeginDisabled();
  ImGui::Button(key);
  ImGui::EndDisabled();
}

// mirrors CenterWidget::createWelcomeWidget() in tools/cabana/detailwidget.cc
void draw_welcome() {
  const ImVec2 size = ImGui::GetContentRegionAvail();
  const float content_height = 50.0f + 4 * (ImGui::GetFrameHeight() + ImGui::GetStyle().ItemSpacing.y);
  ImGui::SetCursorPosY(std::max(0.0f, (size.y - content_height) * 0.5f));
  push_bold_font(50.0f);
  draw_centered_text("CABANA");
  pop_bold_font();
  ImGui::Spacing();
  ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
  draw_centered_text("<- Select a message to view details");
  ImGui::PopStyleColor();
  ImGui::Spacing();
  draw_shortcut_row("Pause", "Space");
  draw_shortcut_row("Help", "F1");
  draw_shortcut_row("WhatsThis", "Shift+F1");
}

}  // namespace

void draw_detail_panel(AppState &app) {
  ImGui::PushStyleColor(ImGuiCol_WindowBg, ImGui::GetStyleColorVec4(ImGuiCol_ChildBg));
  if (ImGui::Begin(CENTER_WINDOW_TITLE, nullptr, ImGuiWindowFlags_NoCollapse)) {
    draw_welcome();
  }
  ImGui::End();
  ImGui::PopStyleColor();
}
