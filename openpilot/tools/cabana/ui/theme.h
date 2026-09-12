#pragma once

#include "imgui.h"
#include "imgui_internal.h"

#include "tools/cabana/core/color.h"

// Palette source: commaai/connect src/{colors,theme}.js at 7091050.
// Dark colors follow connect; light colors use its lightGrey and lightBlue families.
struct Palette {
  ImVec4 text, text_disabled;
  ImVec4 window;   // the background behind panels and docked windows
  ImVec4 surface;  // panels, popups, table bodies: what content is drawn on
  ImVec4 frame, frame_hovered, frame_active;
  ImVec4 button, button_hovered, button_active;
  ImVec4 header, header_hovered, header_active;  // selections
  ImVec4 accent;
  ImVec4 border, separator;
  ImVec4 tab, tab_hovered, table_header;
  ImVec4 grid;
  ImVec4 badge;  // the fill behind the time labels drawn over a chart
};

constexpr ImVec4 rgb(unsigned hex, float alpha = 1.0f) {
  return ImVec4(((hex >> 16) & 255) / 255.0f, ((hex >> 8) & 255) / 255.0f, (hex & 255) / 255.0f, alpha);
}
inline ImVec4 colorRgb(int r, int g, int b, float alpha = 1.0f) {
  return ImVec4(r / 255.0f, g / 255.0f, b / 255.0f, alpha);
}
inline ImU32 toImU32(const CabanaColor &c) { return IM_COL32(c.r, c.g, c.b, c.a); }
inline ImVec4 toImVec4(const CabanaColor &c) { return ImVec4(c.r / 255.0f, c.g / 255.0f, c.b / 255.0f, c.a / 255.0f); }
inline ImU32 withAlpha(ImU32 c, int alpha) { return (c & ~IM_COL32_A_MASK) | ((ImU32)alpha << IM_COL32_A_SHIFT); }

constexpr float UI_FONT_SIZE = 16.0f;

void loadFonts();
void applyTheme(int theme);  // Safe to call at runtime.
bool isDarkTheme();
const Palette &palette();

CabanaColor signalFillColor(const CabanaColor &c);

ImFont *boldFont();
void pushMonoFont(float size = 0.0f);
void popMonoFont();
void pushBoldFont();
void popBoldFont();
void pushLargeFont();
void popLargeFont();
