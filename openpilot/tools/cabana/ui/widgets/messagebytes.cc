#include "tools/cabana/ui/widgets/messagebytes.h"

#include <cmath>

#include <algorithm>
#include <cstdio>

#include "tools/cabana/ui/imgui_util.h"

MessageBytesDelegate::MessageBytesDelegate(bool multiple_lines) : multiple_lines(multiple_lines) {
  for (int i = 0; i < 256; ++i) {
    snprintf(hex_text_table[i], sizeof(hex_text_table[i]), "%02X", i);
  }
}

void MessageBytesDelegate::updateFontMetrics() {
  pushMonoFont();
  // the width of "00 " by the line height (ascent + descent + 1), not the font size
  const ImFontBaked *baked = ImGui::GetFontBaked();
  byte_size = ImVec2(ImGui::CalcTextSize("00 ").x, std::ceil(baked->Ascent) - std::floor(baked->Descent) + 1 + 2);
  popMonoFont();
}

ImVec2 MessageBytesDelegate::sizeForBytes(int n) const {
  int rows = multiple_lines ? std::max(1, n / 8) : 1;  // quirk: n / 8 rounds down, e.g. 12 bytes get 1 row
  const ImVec2 margin = ImGui::GetStyle().CellPadding;  // the margin is one more than the table's cell padding
  return {(n / rows) * byte_size.x + (margin.x + 1) * 2, rows * byte_size.y + (margin.y + 1) * 2};
}

ImVec2 MessageBytesDelegate::sizeHint(const std::vector<uint8_t> *bytes) const {
  return sizeForBytes(bytes ? bytes->size() : 0);
}

void MessageBytesDelegate::paint(ImDrawList *painter, const ImRect &rect, bool selected, bool inactive, const std::string &text,
                                 const std::vector<uint8_t> *bytes, const std::vector<CabanaColor> *colors) const {
  // inactive rows keep their byte colors, gray the text and fade the highlighted text to alpha 100
  ImU32 highlighted_color = highlightedTextColor();
  if (inactive) highlighted_color = (highlighted_color & ~IM_COL32_A_MASK) | ((ImU32)100 << IM_COL32_A_SHIFT);
  const ImU32 text_color = ImGui::GetColorU32(inactive ? ImGuiCol_TextDisabled : ImGuiCol_Text);
  if (!bytes) {
    ImGui::PushStyleColor(ImGuiCol_Text, selected ? highlighted_color : text_color);
    const ImVec2 text_size = ImGui::CalcTextSize(text.c_str());
    const ImVec2 pos(rect.Min.x, rect.Min.y + std::max(0.0f, (rect.GetHeight() - text_size.y) * 0.5f));
    ImGui::RenderTextEllipsis(painter, pos, ImVec2(rect.Max.x, pos.y + text_size.y), rect.Max.x,
                              text.c_str(), nullptr, &text_size);
    ImGui::PopStyleColor();
    return;
  }

  pushMonoFont();
  ImFont *fixed_font = ImGui::GetFont();
  const float font_size = ImGui::GetFontSize();
  const ImU32 text_pen = selected ? highlighted_color : text_color;
  const ImVec2 pt = rect.Min;
  for (int i = 0; i < (int)bytes->size(); ++i) {
    int row = !multiple_lines ? 0 : i / 8;
    int column = !multiple_lines ? i : i % 8;
    const ImVec2 r_min(pt.x + column * byte_size.x, pt.y + row * byte_size.y);
    const ImRect r(r_min, ImVec2(r_min.x + byte_size.x, r_min.y + byte_size.y));

    ImU32 pen = text_pen;
    if (colors && i < (int)colors->size() && (*colors)[i].alpha() > 0) {
      if (selected) {
        pen = ImGui::GetColorU32(ImGuiCol_Text);
        painter->AddRectFilled(r.Min, r.Max, ImGui::GetColorU32(ImGuiCol_WindowBg));
      } else {
        pen = IM_COL32(0, 0, 0, 255);  // QPainter default pen, the Qt delegate never sets it for a colored byte
      }
      painter->AddRectFilled(r.Min, r.Max, toImU32((*colors)[i]));
    }
    // centered in r
    const ImVec2 text_size = ImGui::CalcTextSize(hex_text_table[(*bytes)[i]]);
    const ImVec2 pos(r.Min.x + (byte_size.x - text_size.x) * 0.5f, r.Min.y + (byte_size.y - text_size.y) * 0.5f);
    painter->AddText(fixed_font, font_size, pos, pen, hex_text_table[(*bytes)[i]]);
  }
  popMonoFont();
}
