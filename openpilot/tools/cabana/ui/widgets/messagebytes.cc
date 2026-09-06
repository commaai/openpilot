#include "tools/cabana/ui/widgets/messagebytes.h"

#include <algorithm>
#include <cmath>

#include "tools/cabana/ui/util.h"
#include "tools/cabana/utils/strings.h"

ImVec2 byteCellSize() {
  pushMonoFont();
  // the width of "00 " by the line height (ascent + descent + 1), not the font size
  const ImFontBaked *baked = ImGui::GetFontBaked();
  const ImVec2 size(ImGui::CalcTextSize("00 ").x, std::ceil(baked->Ascent) - std::floor(baked->Descent) + 1 + 2);
  popMonoFont();
  return size;
}

ImVec2 bytesCellSize(int n, bool multiple_lines) {
  const ImVec2 byte_size = byteCellSize();
  const int rows = multiple_lines ? std::max(1, (n + 7) / 8) : 1;
  const int columns = multiple_lines ? std::min(n, 8) : n;
  const ImVec2 margin = ImGui::GetStyle().CellPadding;  // the margin is one more than the table's cell padding
  return {columns * byte_size.x + (margin.x + 1) * 2, rows * byte_size.y + (margin.y + 1) * 2};
}

ImU32 cellTextColor(bool selected, bool inactive) {
  if (selected) return inactive ? withAlpha(highlightedTextColor(), 100) : highlightedTextColor();
  return ImGui::GetColorU32(inactive ? ImGuiCol_TextDisabled : ImGuiCol_Text);
}

void drawTextCell(ImDrawList *dl, const ImRect &rect, const std::string &text, bool selected, bool inactive) {
  drawElidedText(dl, rect, text, cellTextColor(selected, inactive));
}

void drawBytesCell(ImDrawList *dl, const ImRect &rect, const std::vector<uint8_t> &bytes, const std::vector<CabanaColor> *colors,
                   bool selected, bool inactive, bool multiple_lines) {
  const ImU32 text_pen = cellTextColor(selected, inactive);
  const ImVec2 byte_size = byteCellSize();
  pushMonoFont();
  ImFont *font = ImGui::GetFont();
  const float font_size = ImGui::GetFontSize();
  for (int i = 0; i < (int)bytes.size(); ++i) {
    const int row = multiple_lines ? i / 8 : 0;
    const int column = multiple_lines ? i % 8 : i;
    const ImVec2 min(rect.Min.x + column * byte_size.x, rect.Min.y + row * byte_size.y);
    const ImRect r(min, ImVec2(min.x + byte_size.x, min.y + byte_size.y));

    // a colored unselected byte keeps text_pen, like the other cells of the row
    ImU32 pen = text_pen;
    if (colors && i < (int)colors->size() && (*colors)[i].alpha() > 0) {
      if (selected) {
        pen = ImGui::GetColorU32(ImGuiCol_Text);
        dl->AddRectFilled(r.Min, r.Max, ImGui::GetColorU32(ImGuiCol_WindowBg));
      }
      dl->AddRectFilled(r.Min, r.Max, toImU32((*colors)[i]));
    }
    drawText(dl, r, utils::hexByte(bytes[i]), pen, font, font_size);
  }
  popMonoFont();
}
