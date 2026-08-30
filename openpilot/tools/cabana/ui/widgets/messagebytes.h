#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "imgui.h"
#include "imgui_internal.h"
#include "tools/cabana/core/color.h"
#include "tools/cabana/ui/icons.h"

inline ImU32 toImColor(const CabanaColor &color) {
  return IM_COL32(color.r, color.g, color.b, color.a);
}

class MessageBytesDelegate {
public:
  MessageBytesDelegate(bool multiple_lines = false);
  // text cells elide, the bytes cell paints colored hex
  void paint(ImDrawList *painter, const ImRect &rect, bool selected, bool inactive, const std::string &text,
             const std::vector<uint8_t> *bytes = nullptr, const std::vector<CabanaColor> *colors = nullptr) const;
  ImVec2 sizeHint(const std::vector<uint8_t> *bytes) const;
  bool multipleLines() const { return multiple_lines; }
  void setMultipleLines(bool v) { multiple_lines = v; }
  ImVec2 sizeForBytes(int n) const;
  void updateFontMetrics();  // the fonts are only valid inside a frame

private:
  std::array<char[3], 256> hex_text_table;
  ImVec2 byte_size = {};
  bool multiple_lines = false;
  float h_margin = 0, v_margin = 0;
};

// auto-raise button showing a bootstrap glyph with a tooltip
inline bool toolButton(const char *icon, const char *tooltip = "") {
  ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
  bool clicked = ImGui::Button(icon);
  ImGui::PopStyleColor();
  if (tooltip && tooltip[0]) ImGui::SetItemTooltip("%s", tooltip);
  return clicked;
}
