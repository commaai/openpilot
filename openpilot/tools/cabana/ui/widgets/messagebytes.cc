#include "tools/cabana/ui/widgets/messagebytes.h"

#include <algorithm>
#include <cstdio>

#include "tools/cabana/ui/imgui_util.h"

// MessageBytesDelegate

MessageBytesDelegate::MessageBytesDelegate(bool multiple_lines) : multiple_lines(multiple_lines) {
  for (int i = 0; i < 256; ++i) {
    snprintf(hex_text_table[i], sizeof(hex_text_table[i]), "%02X", i);
  }
}

void MessageBytesDelegate::updateFontMetrics() {
  pushMonoFont();
  byte_size = ImGui::CalcTextSize("00 ");
  byte_size.y += 2;
  popMonoFont();
  // PM_FocusFrameHMargin/VMargin + 1: the table's cell padding
  h_margin = ImGui::GetStyle().CellPadding.x;
  v_margin = ImGui::GetStyle().CellPadding.y;
}

ImVec2 MessageBytesDelegate::sizeForBytes(int n) const {
  int rows = multiple_lines ? std::max(1, n / 8) : 1;  // Qt quirk: n / 8 rounds down, e.g. 12 bytes get 1 row
  return {(n / rows) * byte_size.x + h_margin * 2, rows * byte_size.y + v_margin * 2};
}

ImVec2 MessageBytesDelegate::sizeHint(const std::vector<uint8_t> *bytes) const {
  return sizeForBytes(bytes ? bytes->size() : 0);
}

void MessageBytesDelegate::paint(ImDrawList *painter, const ImRect &rect, bool selected, bool inactive, const std::string &text,
                                 const std::vector<uint8_t> *bytes, const std::vector<CabanaColor> *colors) const {
  // the selection background is painted by the row's Selectable

  // the table already applies the cell padding: rect is the item rect
  const ImRect item_rect = rect;
  const ImU32 highlighted_color = ImGui::GetColorU32(ImGuiCol_Text);  // no separate HighlightedText in imgui
  const ImU32 text_color = ImGui::GetColorU32(inactive ? ImGuiCol_TextDisabled : ImGuiCol_Text);
  if (!bytes) {
    ImGui::PushStyleColor(ImGuiCol_Text, selected ? highlighted_color : text_color);
    const ImVec2 text_size = ImGui::CalcTextSize(text.c_str());
    const ImVec2 pos(item_rect.Min.x, item_rect.Min.y + std::max(0.0f, (item_rect.GetHeight() - text_size.y) * 0.5f));
    ImGui::RenderTextEllipsis(painter, pos, ImVec2(item_rect.Max.x, pos.y + text_size.y), item_rect.Max.x,
                              text.c_str(), nullptr, &text_size);
    ImGui::PopStyleColor();
    return;
  }

  // Paint hex column
  pushMonoFont();
  ImFont *fixed_font = ImGui::GetFont();
  const float font_size = ImGui::GetFontSize();
  const ImU32 text_pen = selected ? highlighted_color : text_color;
  const ImVec2 pt = item_rect.Min;
  for (int i = 0; i < (int)bytes->size(); ++i) {
    int row = !multiple_lines ? 0 : i / 8;
    int column = !multiple_lines ? i : i % 8;
    const ImVec2 r_min(pt.x + column * byte_size.x, pt.y + row * byte_size.y);
    const ImRect r(r_min, ImVec2(r_min.x + byte_size.x, r_min.y + byte_size.y));

    ImU32 pen = text_pen;
    if (!inactive && i < (int)colors->size() && (*colors)[i].alpha() > 0) {
      if (selected) {
        pen = ImGui::GetColorU32(ImGuiCol_Text);
        painter->AddRectFilled(r.Min, r.Max, ImGui::GetColorU32(ImGuiCol_WindowBg));
      }
      painter->AddRectFilled(r.Min, r.Max, toImColor((*colors)[i]));
    }
    // drawStaticText: centered in r
    const ImVec2 text_size = ImGui::CalcTextSize(hex_text_table[(*bytes)[i]]);
    const ImVec2 pos(r.Min.x + (byte_size.x - text_size.x) * 0.5f, r.Min.y + (byte_size.y - text_size.y) * 0.5f);
    painter->AddText(fixed_font, font_size, pos, pen, hex_text_table[(*bytes)[i]]);
  }
  popMonoFont();
}
