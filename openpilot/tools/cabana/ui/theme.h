#pragma once

#include "imgui.h"
#include "imgui_internal.h"

#include "tools/cabana/core/color.h"

struct Palette {
  ImVec4 text, text_disabled, text_inactive, text_selected;
  ImVec4 window;   // the background behind panels and docked windows
  ImVec4 surface;  // panels, popups, table bodies: what content is drawn on
  ImVec4 frame, frame_hovered, frame_active;
  ImVec4 button, button_hovered, button_active;
  ImVec4 header, header_hovered, header_active;  // selected fill; unselected hover/press fills
  ImVec4 accent;
  ImVec4 border, separator, scrollbar_grab;
  ImVec4 slider_track;
  ImVec4 tab, tab_hovered, tab_selected, table_header;
  ImVec4 grid;
  ImVec4 badge;  // the fill behind the time labels drawn over a chart
  float sparkline_saturation, sparkline_value;  // HSV multipliers for signal colors
};

constexpr ImVec4 rgb(unsigned hex, float alpha = 1.0f) {
  return ImVec4(((hex >> 16) & 255) / 255.0f, ((hex >> 8) & 255) / 255.0f, (hex & 255) / 255.0f, alpha);
}
inline ImVec4 colorRgb(int r, int g, int b, float alpha = 1.0f) {
  return ImVec4(r / 255.0f, g / 255.0f, b / 255.0f, alpha);
}
inline ImU32 toImU32(const CabanaColor &c) { return IM_COL32(c.r, c.g, c.b, c.a); }
inline CabanaColor fromImVec4(const ImVec4 &c) {
  return {static_cast<uint8_t>(std::lround(c.x * 255)), static_cast<uint8_t>(std::lround(c.y * 255)),
          static_cast<uint8_t>(std::lround(c.z * 255)), static_cast<uint8_t>(std::lround(c.w * 255))};
}
inline ImVec4 toImVec4(const CabanaColor &c) { return ImVec4(c.r / 255.0f, c.g / 255.0f, c.b / 255.0f, c.a / 255.0f); }
inline ImU32 withAlpha(ImU32 c, int alpha) { return (c & ~IM_COL32_A_MASK) | ((ImU32)alpha << IM_COL32_A_SHIFT); }

// Logical pixels. External control gaps are equal on both axes; inner spacing
// is reserved for parts of one control (icon/label, checkbox/label, dropdown arrow).
namespace spacing {
constexpr float CONTROL = 8.0f;
constexpr float INNER = 4.0f;
constexpr float DIALOG_BUTTON_MIN_WIDTH = 80.0f;
}  // namespace spacing

constexpr float UI_FONT_SIZE = 16.0f;

void loadFonts();
void applyTheme(int theme);  // Safe to call at runtime.
const Palette &palette();
CabanaColor contrastColor(CabanaColor color, const CabanaColor &background, double target = 4.5);
CabanaColor byteColor(const CabanaColor &color);
CabanaColor signalFill(const CabanaColor &color, bool defined = true);
CabanaColor signalHighlight(CabanaColor color);
CabanaColor signalOutline(CabanaColor color, bool hovered = false);
CabanaColor graphicColor(const CabanaColor &color, const ImVec4 &background = palette().surface);
CabanaColor sparklineColor(const CabanaColor &color);

ImFont *boldFont();
void pushMonoFont(float size = 0.0f);
void popMonoFont();
void pushBoldFont();
void popBoldFont();
void pushLargeFont();
void popLargeFont();
