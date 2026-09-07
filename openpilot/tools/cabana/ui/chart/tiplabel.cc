#define IMGUI_DEFINE_MATH_OPERATORS  // ImVec2 arithmetic, must precede imgui.h
#include "tools/cabana/ui/chart/tiplabel.h"

#include <algorithm>
#include <cmath>

#include "tools/cabana/ui/util.h"

ImVec2 TipLabel::layoutLines(ImDrawList *p, const ImVec2 &origin, ImU32 fg) const {
  const float font_size = ImGui::GetFontSize();
  const float line_height = std::ceil(ImGui::GetTextLineHeight() + 2);
  const float marker = std::floor(markerSize());
  const float gap = 8;
  const float width = column_widths_[0] + column_widths_[1] + column_widths_[2] + column_widths_[3] + gap * 3;
  const ImU32 muted = ImGui::GetColorU32(ImGuiCol_TextDisabled);
  float y = std::round(origin.y);
  auto draw = [&](float x, const std::string &text, ImU32 color) {
    if (p) p->AddText(ImVec2(std::round(x), y), color, text.c_str());
  };
  const char *heading = !text_.empty() && !text_[0].has_marker ? text_[0].name.c_str() : "Signal";
  const char *headers[] = {heading, "Value", "Min", "Max"};
  float x = origin.x;
  for (int i = 0; i < 4; ++i) {
    draw(i ? x + column_widths_[i] - ImGui::CalcTextSize(headers[i]).x : x, headers[i], muted);
    x += column_widths_[i] + gap;
  }
  y += line_height;
  if (p) p->AddLine(ImVec2(origin.x, y - 2), ImVec2(origin.x + width, y - 2), ImGui::GetColorU32(ImGuiCol_Border));
  for (const auto &line : text_) {
    if (!line.has_marker) continue;
    if (p) {
      const ImVec2 marker_pos(std::round(origin.x), std::round(y + (font_size - marker) * 0.5f));
      p->AddRectFilled(marker_pos, marker_pos + ImVec2(marker, marker), line.marker);
    }
    if (p) drawElidedText(p, ImRect(ImVec2(origin.x + marker + 6, y),
                                  ImVec2(origin.x + column_widths_[0], y + font_size)), line.name, fg);
    x = origin.x + column_widths_[0] + gap;
    pushMonoFont(font_size);
    const std::string *values[] = {&line.value, &line.min, &line.max};
    for (int i = 0; i < 3; ++i) {
      draw(x + column_widths_[i + 1] - ImGui::CalcTextSize(values[i]->c_str()).x, *values[i], i == 0 ? fg : muted);
      x += column_widths_[i + 1] + gap;
    }
    popMonoFont();
    y += line_height;
  }
  return ImVec2(width, y - origin.y);
}

ImVec2 TipLabel::sizeHint() const {
  return layoutLines(nullptr, ImVec2(0, 0), 0) + ImVec2(MARGIN * 2, MARGIN * 2);
}

void TipLabel::showText(const ImVec2 &pt, const std::vector<TipLine> &text, const ImRect &rect) {
  bool same_signals = text.size() == text_.size();
  for (size_t i = 1; same_signals && i < text.size(); ++i) same_signals = text[i].name == text_[i].name;
  if (!same_signals) column_widths_ = {};
  text_ = text;
  anchor_ = pt;
  area_ = rect;
  visible_ = !text_.empty();
}

void TipLabel::updateLayout() {
  // Playback notifications can update the text outside an ImGui window scope.
  // Measure and position it only while the owning chart is being drawn.
  column_widths_[0] = std::max(column_widths_[0], ImGui::CalcTextSize(text_.empty() ? "Signal" : text_[0].name.c_str()).x);
  for (const auto &line : text_) {
    if (line.has_marker) column_widths_[0] = std::max(column_widths_[0], std::ceil(markerSize() + 6 + ImGui::CalcTextSize(line.name.c_str()).x));
  }
  pushMonoFont(ImGui::GetFontSize());
  const float number_width = std::ceil(ImGui::CalcTextSize("-0.00000").x);
  for (int i = 1; i < 4; ++i) column_widths_[i] = std::max(column_widths_[i], number_width);
  for (const auto &line : text_) {
    const std::string *values[] = {&line.value, &line.min, &line.max};
    for (int i = 0; i < 3; ++i) column_widths_[i + 1] = std::max(column_widths_[i + 1], std::ceil(ImGui::CalcTextSize(values[i]->c_str()).x));
  }
  popMonoFont();
  const ImGuiViewport *viewport = ImGui::GetWindowViewport();
  const ImRect bounds(viewport->WorkPos, viewport->WorkPos + viewport->WorkSize);
  const float numeric_width = column_widths_[1] + column_widths_[2] + column_widths_[3] + 24 + MARGIN * 2 + 1;
  column_widths_[0] = std::min(column_widths_[0], std::max(40.0f, std::min(ImGui::GetFontSize() * 16, bounds.GetWidth() - numeric_width)));
  if (!text_.empty()) {
    ImVec2 extra(1, 1);
    size_ = sizeHint() + extra;
    ImVec2 tip_pos(anchor_.x + 8, area_.Min.y + 2);
    if (anchor_.x >= area_.GetCenter().x) {
      tip_pos.x = anchor_.x - size_.x - 8;
    }
    tip_pos.x = std::clamp(tip_pos.x, bounds.Min.x, std::max(bounds.Min.x, bounds.Max.x - size_.x));
    tip_pos.y = std::clamp(tip_pos.y, bounds.Min.y, std::max(bounds.Min.y, bounds.Max.y - size_.y));
    pos_ = tip_pos;
    visible_ = true;
    return;
  }
  visible_ = false;
}

void TipLabel::draw() {
  if (!visible_) return;
  updateLayout();

  ImDrawList *p = ImGui::GetForegroundDrawList();
  // filled panel with a 1px frame
  p->AddRectFilled(pos_, pos_ + size_, ImGui::GetColorU32(ImGuiCol_PopupBg), ImGui::GetStyle().PopupRounding);
  p->AddRect(pos_, pos_ + size_, ImGui::GetColorU32(ImGuiCol_Border), ImGui::GetStyle().PopupRounding);
  layoutLines(p, pos_ + ImVec2(MARGIN, MARGIN), ImGui::GetColorU32(ImGuiCol_Text));
}
