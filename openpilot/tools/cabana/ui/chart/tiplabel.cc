#define IMGUI_DEFINE_MATH_OPERATORS  // ImVec2 arithmetic, must precede imgui.h
#include "tools/cabana/ui/chart/tiplabel.h"

#include <algorithm>
#include <cfloat>

#include "tools/cabana/ui/util.h"

ImVec2 TipLabel::layoutLines(ImDrawList *p, const ImVec2 &origin, ImU32 fg) const {
  ImFont *bold = boldFont();
  const float font_size = ImGui::GetFontSize();
  const float line_height = ImGui::GetTextLineHeight();
  ImVec2 size(0, 0);
  float y = origin.y;
  for (const auto &line : text_) {
    float x = origin.x;
    if (line.has_marker) {
      if (p) drawColorMarker(p, ImVec2(x, y), line.marker);
      x += markerSize() + 4;
    }
    if (p) p->AddText(ImVec2(x, y), fg, line.name.c_str());
    x += ImGui::CalcTextSize(line.name.c_str()).x;
    if (!line.bold.empty()) {
      if (p) p->AddText(bold, font_size, ImVec2(x, y), fg, line.bold.c_str());
      x += bold->CalcTextSizeA(font_size, FLT_MAX, 0.0f, line.bold.c_str()).x;
    }
    if (p) p->AddText(ImVec2(x, y), fg, line.rest.c_str());
    x += ImGui::CalcTextSize(line.rest.c_str()).x;
    size.x = std::max(size.x, x - origin.x);
    y += line_height;
  }
  size.y = y - origin.y;
  return size;
}

ImVec2 TipLabel::sizeHint() const {
  return layoutLines(nullptr, ImVec2(0, 0), 0) + ImVec2(MARGIN * 2, MARGIN * 2);
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

void TipLabel::draw() {
  if (!visible_) return;

  ImDrawList *p = ImGui::GetForegroundDrawList();
  const bool dark = isDarkTheme();
  const ImU32 bg = dark ? ImGui::GetColorU32(ImGuiCol_PopupBg) : ImGui::GetColorU32(ImGuiCol_ChildBg);
  const ImU32 fg = dark ? ImGui::GetColorU32(ImGuiCol_Text) : IM_COL32(0x40, 0x40, 0x44, 0xff);
  // filled panel with a 1px frame
  p->AddRectFilled(pos_, pos_ + size_, bg);
  p->AddRect(pos_, pos_ + size_, ImGui::GetColorU32(ImGuiCol_Border));
  layoutLines(p, pos_ + ImVec2(MARGIN, MARGIN), fg);
}
