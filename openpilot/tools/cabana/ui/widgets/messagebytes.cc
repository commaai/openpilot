#include "tools/cabana/ui/widgets/messagebytes.h"

#include <algorithm>
#include <cmath>

#include "tools/cabana/ui/util.h"
#include "tools/cabana/utils/strings.h"

void MessageBytesDelegate::updateFontMetrics() {
  pushMonoFont();
  // the width of "00 " by the line height (ascent + descent + 1), not the font size
  const ImFontBaked *baked = ImGui::GetFontBaked();
  byte_size_ = ImVec2(ImGui::CalcTextSize("00 ").x, std::ceil(baked->Ascent) - std::floor(baked->Descent) + 1 + 2);
  popMonoFont();
}

ImVec2 MessageBytesDelegate::sizeForBytes(int n) const {
  const int rows = multiple_lines_ ? std::max(1, (n + 7) / 8) : 1;
  const int columns = multiple_lines_ ? std::min(n, 8) : n;
  const ImVec2 margin = ImGui::GetStyle().CellPadding;  // the margin is one more than the table's cell padding
  return {columns * byte_size_.x + (margin.x + 1) * 2, rows * byte_size_.y + (margin.y + 1) * 2};
}

ImVec2 MessageBytesDelegate::sizeHint(const std::vector<uint8_t> *bytes) const {
  return sizeForBytes(bytes ? bytes->size() : 0);
}

void MessageBytesDelegate::paint(ImDrawList *painter, const ImRect &rect, bool selected, bool inactive, const std::string &text,
                                 const std::vector<uint8_t> *bytes, const std::vector<CabanaColor> *colors) const {
  // inactive rows keep their byte colors, gray the text and fade the highlighted text to alpha 100
  ImU32 highlighted_color = highlightedTextColor();
  if (inactive) highlighted_color = withAlpha(highlighted_color, 100);
  const ImU32 text_color = ImGui::GetColorU32(inactive ? ImGuiCol_TextDisabled : ImGuiCol_Text);
  const ImU32 text_pen = selected ? highlighted_color : text_color;
  if (!bytes) {
    drawElidedText(painter, rect, text, text_pen);
    return;
  }

  pushMonoFont();
  ImFont *fixed_font = ImGui::GetFont();
  const float font_size = ImGui::GetFontSize();
  const ImVec2 pt = rect.Min;
  for (int i = 0; i < (int)bytes->size(); ++i) {
    int row = !multiple_lines_ ? 0 : i / 8;
    int column = !multiple_lines_ ? i : i % 8;
    const ImVec2 r_min(pt.x + column * byte_size_.x, pt.y + row * byte_size_.y);
    const ImRect r(r_min, ImVec2(r_min.x + byte_size_.x, r_min.y + byte_size_.y));

    // a colored unselected byte keeps text_pen: the Qt delegate leaves the pen the row's other cells set
    ImU32 pen = text_pen;
    if (colors && i < (int)colors->size() && (*colors)[i].alpha() > 0) {
      if (selected) {
        pen = ImGui::GetColorU32(ImGuiCol_Text);
        painter->AddRectFilled(r.Min, r.Max, ImGui::GetColorU32(ImGuiCol_WindowBg));
      }
      painter->AddRectFilled(r.Min, r.Max, toImU32((*colors)[i]));
    }
    drawText(painter, r, utils::hexByte((*bytes)[i]), pen, fixed_font, font_size);
  }
  popMonoFont();
}
