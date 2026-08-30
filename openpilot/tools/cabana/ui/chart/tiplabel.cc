#define IMGUI_DEFINE_MATH_OPERATORS  // ImVec2 arithmetic, must precede imgui.h
#include "tools/cabana/ui/chart/tiplabel.h"

#include <algorithm>
#include <utility>

#include "tools/cabana/core/settings.h"
#include "tools/cabana/settings.h"
#include "tools/cabana/ui/imgui_util.h"

ImVec2 TipLabel::sizeHint() const {
  ImVec2 size(0, 0);
  const float marker = ImGui::GetTextLineHeight() - 4;
  for (const auto &line : text_) {
    float w = 0;
    if (line.has_marker) w += marker + 4;
    w += ImGui::CalcTextSize(line.name.c_str()).x;
    if (!line.bold.empty()) {
      pushBoldFont();
      w += ImGui::CalcTextSize(line.bold.c_str()).x;
      popBoldFont();
    }
    w += ImGui::CalcTextSize(line.rest.c_str()).x;
    size.x = std::max(size.x, w);
    size.y += ImGui::GetTextLineHeight();
  }
  return ImVec2(size.x + margin_ * 2, size.y + margin_ * 2);
}

void TipLabel::showText(const ImVec2 &pt, const std::vector<TipLine> &text, const ImRect &rect) {
  text_ = text;
  if (!text_.empty()) {
    ImVec2 extra(1, 1);
    size_ = sizeHint() + extra;
    ImVec2 tip_pos(pt.x + 8, rect.Min.y + 2);
    if (tip_pos.x + size_.x >= rect.Max.x) {
      tip_pos.x = pt.x - size_.x - 8;
    }
    if (rect.Contains(ImRect(tip_pos, tip_pos + size_))) {
      pos_ = tip_pos;
      visible_ = true;
      return;
    }
  }
  visible_ = false;
}

void TipLabel::paintEvent() {
  if (!visible_) return;

  ImDrawList *p = ImGui::GetForegroundDrawList();
  const bool dark = isDarkTheme();
  const ImU32 bg = dark ? ImGui::GetColorU32(ImGuiCol_PopupBg) : ImGui::GetColorU32(ImGuiCol_ChildBg);
  const ImU32 fg = dark ? ImGui::GetColorU32(ImGuiCol_Text) : IM_COL32(0x40, 0x40, 0x44, 0xff);
  // filled panel with a 1px frame
  p->AddRectFilled(pos_, pos_ + size_, bg);
  p->AddRect(pos_, pos_ + size_, ImGui::GetColorU32(ImGuiCol_Border));

  const float line_height = ImGui::GetTextLineHeight();
  const float marker = line_height - 4;
  float y = pos_.y + margin_;
  for (const auto &line : text_) {
    float x = pos_.x + margin_;
    if (line.has_marker) {
      p->AddRectFilled(ImVec2(x, y + 2), ImVec2(x + marker, y + 2 + marker), line.marker);
      x += marker + 4;
    }
    p->AddText(ImVec2(x, y), fg, line.name.c_str());
    x += ImGui::CalcTextSize(line.name.c_str()).x;
    if (!line.bold.empty()) {
      pushBoldFont();
      p->AddText(ImGui::GetFont(), ImGui::GetFontSize(), ImVec2(x, y), fg, line.bold.c_str());
      x += ImGui::CalcTextSize(line.bold.c_str()).x;
      popBoldFont();
    }
    p->AddText(ImVec2(x, y), fg, line.rest.c_str());
    y += line_height;
  }
}
