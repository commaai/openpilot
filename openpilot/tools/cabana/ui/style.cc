#include "tools/cabana/ui/app.h"

#include <algorithm>
#include <cmath>
#include <filesystem>

#include "implot.h"
#include "tools/cabana/core/settings.h"
#include "tools/cabana/settings.h"
#include "tools/cabana/ui/util.h"
#include "tools/cabana/utils/util.h"

namespace fs = std::filesystem;

namespace {
bool g_dark = false;
ImFont *g_ui_font = nullptr;
ImFont *g_bold_font = nullptr;
ImFont *g_mono_font = nullptr;
ImFont *g_large_font = nullptr;

void addIconFont(float size, ImFont *base) {
  ImFontConfig cfg;
  cfg.MergeMode = base != nullptr;
  cfg.GlyphMinAdvanceX = size;
  if (base != nullptr) {
    ImFontBaked *baked = base->GetFontBaked(size);
    const float center = baked != nullptr ? (baked->Ascent + baked->Descent) * 0.5f : size * 0.5f;
    cfg.GlyphOffset.y = std::round(size * 0.5f - center);
  }
  static const ImWchar ranges[] = {0xF000, 0xF8FF, 0};
  ImGui::GetIO().Fonts->AddFontFromFileTTF(BOOTSTRAP_ICONS_TTF, size, &cfg, ranges);
}

ImFont *addFont(const fs::path &path, float size) {
  ImFontConfig cfg;
  cfg.OversampleH = 2;
  cfg.OversampleV = 2;
  ImFont *font = ImGui::GetIO().Fonts->AddFontFromFileTTF(path.c_str(), size, &cfg);
  if (font != nullptr) addIconFont(size, font);
  return font;
}
}  // namespace

void loadFonts() {
  ImGuiIO &io = ImGui::GetIO();
  const fs::path fonts = fs::path(CABANA_FONTS_DIR);
  g_ui_font = addFont(fonts / "Inter-Regular.ttf", 16.0f);
  g_bold_font = addFont(fonts / "Inter-SemiBold.ttf", 16.0f);
  g_mono_font = addFont(fonts / "JetBrainsMono-Medium.ttf", 15.0f);
  g_large_font = addFont(fonts / "Inter-Bold.ttf", 50.0f);
  if (g_ui_font != nullptr) io.FontDefault = g_ui_font;
  if (g_bold_font == nullptr) g_bold_font = g_ui_font;
  if (g_mono_font == nullptr) g_mono_font = g_ui_font;
  if (g_large_font == nullptr) g_large_font = g_bold_font;
}

void applyTheme(int theme) {
  const bool dark = theme == DARK_THEME;
  g_dark = dark;
  if (dark) {
    ImGui::StyleColorsDark();
    ImPlot::StyleColorsDark();
  } else {
    ImGui::StyleColorsLight();
    ImPlot::StyleColorsLight();
  }

  ImGuiStyle &style = ImGui::GetStyle();
  style.WindowRounding = 0.0f;
  style.ChildRounding = 0.0f;
  style.PopupRounding = 0.0f;
  style.FrameRounding = 2.0f;
  style.GrabRounding = 2.0f;
  style.ScrollbarRounding = 2.0f;
  style.TabRounding = 2.0f;
  style.WindowBorderSize = 1.0f;
  style.FrameBorderSize = 1.0f;
  style.TabBorderSize = 1.0f;
  style.WindowPadding = ImVec2(8.0f, 7.0f);
  style.FramePadding = ImVec2(6.0f, 3.0f);
  style.ItemSpacing = ImVec2(8.0f, 5.0f);
  style.ScrollbarSize = 14.0f;
  style.GrabMinSize = 13.0f;

  auto c = [](const CabanaColor &col, float a = 1.0f) { return colorRgb(col.r, col.g, col.b, a); };
  ImVec4 *colors = style.Colors;
  if (dark) {
    // the low contrast Darcula grays are opened up: text and outlines sit further from the window and
    // base grays
    const ImVec4 highlight = c(DarkTheme::highlight);
    const ImVec4 outline = colorRgb(0x5a, 0x5d, 0x60);
    colors[ImGuiCol_WindowBg] = c(DarkTheme::window);
    colors[ImGuiCol_ChildBg] = c(DarkTheme::base);
    colors[ImGuiCol_PopupBg] = c(DarkTheme::base);
    colors[ImGuiCol_MenuBarBg] = c(DarkTheme::window);
    colors[ImGuiCol_DockingEmptyBg] = c(DarkTheme::window);
    colors[ImGuiCol_Text] = colorRgb(0xdc, 0xdc, 0xdc);
    colors[ImGuiCol_TextDisabled] = colorRgb(0x8c, 0x8c, 0x8c);
    colors[ImGuiCol_Border] = outline;
    colors[ImGuiCol_BorderShadow] = colorRgb(0, 0, 0, 0.0f);
    colors[ImGuiCol_FrameBg] = colorRgb(0x2e, 0x30, 0x32);  // darker than the base so fields read as sunken
    colors[ImGuiCol_FrameBgHovered] = colorRgb(0x3a, 0x3d, 0x40);
    colors[ImGuiCol_FrameBgActive] = colorRgb(0x45, 0x48, 0x4b);
    colors[ImGuiCol_Button] = c(DarkTheme::button);
    colors[ImGuiCol_ButtonHovered] = colorRgb(0x52, 0x56, 0x59);
    colors[ImGuiCol_ButtonActive] = colorRgb(0x2b, 0x2d, 0x30);
    colors[ImGuiCol_Header] = highlight;
    colors[ImGuiCol_HeaderHovered] = c(DarkTheme::highlight, 0.8f);
    colors[ImGuiCol_HeaderActive] = highlight;
    colors[ImGuiCol_CheckMark] = c(DarkTheme::bright_text);
    colors[ImGuiCol_SliderGrab] = colorRgb(0x8f, 0x92, 0x95);
    colors[ImGuiCol_SliderGrabActive] = colorRgb(0xa8, 0xab, 0xae);
    colors[ImGuiCol_ScrollbarBg] = c(DarkTheme::window);
    colors[ImGuiCol_ScrollbarGrab] = colorRgb(0x70, 0x73, 0x76);
    colors[ImGuiCol_ScrollbarGrabHovered] = colorRgb(0x85, 0x88, 0x8b);
    colors[ImGuiCol_ScrollbarGrabActive] = c(DarkTheme::light);
    colors[ImGuiCol_Separator] = outline;
    colors[ImGuiCol_SeparatorHovered] = c(DarkTheme::highlight, 0.6f);
    colors[ImGuiCol_SeparatorActive] = highlight;
    colors[ImGuiCol_ResizeGrip] = colorRgb(0, 0, 0, 0.0f);
    colors[ImGuiCol_ResizeGripHovered] = c(DarkTheme::highlight, 0.6f);
    colors[ImGuiCol_ResizeGripActive] = highlight;
    colors[ImGuiCol_Tab] = c(DarkTheme::window);
    colors[ImGuiCol_TabHovered] = colorRgb(0x4b, 0x4e, 0x52);
    colors[ImGuiCol_TabSelected] = c(DarkTheme::base);
    colors[ImGuiCol_TabSelectedOverline] = highlight;
    colors[ImGuiCol_TabDimmed] = c(DarkTheme::window);
    colors[ImGuiCol_TabDimmedSelected] = c(DarkTheme::base);
    colors[ImGuiCol_TabDimmedSelectedOverline] = colorRgb(0, 0, 0, 0.0f);
    colors[ImGuiCol_TitleBg] = c(DarkTheme::window);
    colors[ImGuiCol_TitleBgActive] = c(DarkTheme::window);
    colors[ImGuiCol_TitleBgCollapsed] = c(DarkTheme::window);
    colors[ImGuiCol_TableHeaderBg] = c(DarkTheme::window);
    colors[ImGuiCol_TableBorderStrong] = outline;
    colors[ImGuiCol_TableBorderLight] = colorRgb(0x23, 0x26, 0x28);  // darker than the cells, like the qt grid
    colors[ImGuiCol_TableRowBg] = colorRgb(0, 0, 0, 0.0f);
    colors[ImGuiCol_TableRowBgAlt] = colorRgb(0xff, 0xff, 0xff, 0.06f);
    colors[ImGuiCol_TextSelectedBg] = c(DarkTheme::highlight, 0.6f);
    colors[ImGuiCol_DockingPreview] = c(DarkTheme::highlight, 0.5f);
    colors[ImGuiCol_NavCursor] = highlight;
    colors[ImGuiCol_PlotLines] = c(DarkTheme::text);
    colors[ImGuiCol_PlotHistogram] = highlight;
    colors[ImGuiCol_DragDropTarget] = highlight;
  } else {
    const ImVec4 window = colorRgb(0xef, 0xef, 0xef);
    const ImVec4 base = colorRgb(0xff, 0xff, 0xff);
    const ImVec4 outline = colorRgb(0xb9, 0xb9, 0xb9);
    const ImVec4 highlight = colorRgb(0x30, 0x8c, 0xc6);
    colors[ImGuiCol_WindowBg] = window;
    colors[ImGuiCol_ChildBg] = base;
    colors[ImGuiCol_PopupBg] = colorRgb(0xfb, 0xfb, 0xfb);
    colors[ImGuiCol_MenuBarBg] = window;
    colors[ImGuiCol_DockingEmptyBg] = window;
    colors[ImGuiCol_Text] = colorRgb(0x00, 0x00, 0x00);
    colors[ImGuiCol_TextDisabled] = colorRgb(0xbe, 0xbe, 0xbe);
    colors[ImGuiCol_Border] = outline;
    colors[ImGuiCol_BorderShadow] = colorRgb(0, 0, 0, 0.0f);
    colors[ImGuiCol_FrameBg] = base;
    colors[ImGuiCol_FrameBgHovered] = colorRgb(0xf7, 0xf7, 0xf7);
    colors[ImGuiCol_FrameBgActive] = colorRgb(0xef, 0xef, 0xef);
    colors[ImGuiCol_Button] = colorRgb(0xf3, 0xf3, 0xf3);
    colors[ImGuiCol_ButtonHovered] = colorRgb(0xf9, 0xf9, 0xf9);
    colors[ImGuiCol_ButtonActive] = colorRgb(0xdc, 0xdc, 0xdc);
    colors[ImGuiCol_Header] = highlight;
    colors[ImGuiCol_HeaderHovered] = colorRgb(0x30, 0x8c, 0xc6, 0.8f);
    colors[ImGuiCol_HeaderActive] = highlight;
    colors[ImGuiCol_CheckMark] = colorRgb(0x3b, 0x3b, 0x3b);
    colors[ImGuiCol_SliderGrab] = colorRgb(0xd8, 0xd8, 0xd8);
    colors[ImGuiCol_SliderGrabActive] = colorRgb(0xc4, 0xc4, 0xc4);
    colors[ImGuiCol_ScrollbarBg] = window;
    colors[ImGuiCol_ScrollbarGrab] = colorRgb(0xc8, 0xc8, 0xc8);
    colors[ImGuiCol_ScrollbarGrabHovered] = colorRgb(0xb4, 0xb4, 0xb4);
    colors[ImGuiCol_ScrollbarGrabActive] = colorRgb(0xa0, 0xa0, 0xa0);
    colors[ImGuiCol_Separator] = outline;
    colors[ImGuiCol_SeparatorHovered] = colorRgb(0x30, 0x8c, 0xc6, 0.6f);
    colors[ImGuiCol_SeparatorActive] = highlight;
    colors[ImGuiCol_ResizeGrip] = colorRgb(0, 0, 0, 0.0f);
    colors[ImGuiCol_ResizeGripHovered] = colorRgb(0x30, 0x8c, 0xc6, 0.6f);
    colors[ImGuiCol_ResizeGripActive] = highlight;
    colors[ImGuiCol_Tab] = colorRgb(0xe2, 0xe2, 0xe2);
    colors[ImGuiCol_TabHovered] = colorRgb(0xf5, 0xf5, 0xf5);
    colors[ImGuiCol_TabSelected] = base;
    colors[ImGuiCol_TabSelectedOverline] = highlight;
    colors[ImGuiCol_TabDimmed] = colorRgb(0xe2, 0xe2, 0xe2);
    colors[ImGuiCol_TabDimmedSelected] = base;
    colors[ImGuiCol_TabDimmedSelectedOverline] = colorRgb(0, 0, 0, 0.0f);
    colors[ImGuiCol_TitleBg] = window;
    colors[ImGuiCol_TitleBgActive] = window;
    colors[ImGuiCol_TitleBgCollapsed] = window;
    colors[ImGuiCol_TableHeaderBg] = colorRgb(0xf2, 0xf2, 0xf2);
    colors[ImGuiCol_TableBorderStrong] = outline;
    colors[ImGuiCol_TableBorderLight] = colorRgb(0xd8, 0xd8, 0xd8);
    colors[ImGuiCol_TableRowBg] = colorRgb(0, 0, 0, 0.0f);
    colors[ImGuiCol_TableRowBgAlt] = colorRgb(0, 0, 0, 0.03f);
    colors[ImGuiCol_TextSelectedBg] = colorRgb(0x30, 0x8c, 0xc6, 0.35f);
    colors[ImGuiCol_DockingPreview] = colorRgb(0x30, 0x8c, 0xc6, 0.5f);
    colors[ImGuiCol_NavCursor] = highlight;
    colors[ImGuiCol_PlotLines] = colorRgb(0x3b, 0x3b, 0x3b);
    colors[ImGuiCol_PlotHistogram] = highlight;
    colors[ImGuiCol_DragDropTarget] = highlight;
  }
  // imgui fades the modal dim in over several frames, which reads as the dialog lagging
  colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0, 0, 0, 0);
  colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0, 0, 0, 0);
}

bool isDarkTheme() { return g_dark; }

ImU32 highlightedTextColor() {
  return g_dark ? IM_COL32(DarkTheme::window_text.r, DarkTheme::window_text.g, DarkTheme::window_text.b, 255)
                : IM_COL32(255, 255, 255, 255);
}

ImU32 paletteBrightText() {
  return g_dark ? IM_COL32(DarkTheme::bright_text.r, DarkTheme::bright_text.g, DarkTheme::bright_text.b, 255)
                : IM_COL32(255, 255, 255, 255);
}

void drawSliderHandle(ImDrawList *p, const ImRect &r) {
  const bool dark = isDarkTheme();
  const ImU32 top = dark ? IM_COL32(0x3e, 0x41, 0x43, 255) : IM_COL32(255, 255, 255, 255);
  const ImU32 bottom = dark ? IM_COL32(0x39, 0x3c, 0x3e, 255) : IM_COL32(0xf0, 0xf0, 0xf0, 255);
  // the top/left edge is one step lighter than the bottom/right edge
  const ImU32 outline_top = dark ? IM_COL32(0xa3, 0xa3, 0xa3, 255) : IM_COL32(0xab, 0xab, 0xab, 255);
  const ImU32 outline_bottom = dark ? IM_COL32(0x9c, 0x9c, 0x9c, 255) : IM_COL32(0xa4, 0xa4, 0xa4, 255);
  p->AddRectFilled(r.Min, r.Max, top, 2.0f);
  p->AddRectFilled(ImVec2(r.Min.x, r.GetCenter().y), r.Max, bottom, 2.0f, ImDrawFlags_RoundCornersBottom);
  p->AddRect(r.Min, r.Max, outline_bottom, 2.0f, 0, 1.0f);
  // the straight edges are drawn as crisp 1 px rects: an antialiased outline washes out to a much lighter grey
  const float c = 2.0f;  // corner radius
  p->AddRectFilled(ImVec2(r.Min.x + c, r.Min.y), ImVec2(r.Max.x - c, r.Min.y + 1.0f), outline_top);
  p->AddRectFilled(ImVec2(r.Min.x, r.Min.y + c), ImVec2(r.Min.x + 1.0f, r.Max.y - c), outline_top);
  p->AddRectFilled(ImVec2(r.Min.x + c, r.Max.y - 1.0f), ImVec2(r.Max.x - c, r.Max.y), outline_bottom);
  p->AddRectFilled(ImVec2(r.Max.x - 1.0f, r.Min.y + c), ImVec2(r.Max.x, r.Max.y - c), outline_bottom);
}

bool fusionSliderInt(const char *label, int *v, int min, int max, float width) {
  // a grey groove over the full width with the part left of the handle filled, and a 13x13 handle on top
  const ImU32 groove_col = isDarkTheme() ? IM_COL32(0x2a, 0x2c, 0x2e, 255) : IM_COL32(0xc4, 0xc4, 0xc4, 255);
  const ImU32 fill_col = ImGui::GetColorU32(ImGuiCol_Header);
  ImGui::PushStyleColor(ImGuiCol_FrameBg, IM_COL32_BLACK_TRANS);
  ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, IM_COL32_BLACK_TRANS);
  ImGui::PushStyleColor(ImGuiCol_FrameBgActive, IM_COL32_BLACK_TRANS);
  ImGui::PushStyleColor(ImGuiCol_SliderGrab, IM_COL32_BLACK_TRANS);
  ImGui::PushStyleColor(ImGuiCol_SliderGrabActive, IM_COL32_BLACK_TRANS);
  ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);  // the slider has no frame
  ImGui::SetNextItemWidth(width);
  bool changed = ImGui::SliderInt(label, v, min, max, "", ImGuiSliderFlags_NoInput);
  ImGui::PopStyleVar();
  ImGui::PopStyleColor(5);

  const ImVec2 bb_min = ImGui::GetItemRectMin(), bb_max = ImGui::GetItemRectMax();
  const float cy = (bb_min.y + bb_max.y) * 0.5f;
  const float groove_h = SLIDER_THICKNESS * 0.5f;
  const float handle_h = std::min(SLIDER_THICKNESS, bb_max.y - bb_min.y);
  const float x0 = bb_min.x + SLIDER_LENGTH * 0.5f, x1 = bb_max.x - SLIDER_LENGTH * 0.5f;
  const float t = max > min ? (float)(*v - min) / (float)(max - min) : 0.0f;
  const float hx = x0 + (x1 - x0) * t;
  ImDrawList *dl = ImGui::GetWindowDrawList();
  const float groove_y0 = cy - groove_h * 0.5f, groove_y1 = cy + groove_h * 0.5f;
  dl->AddRectFilled(ImVec2(bb_min.x, groove_y0), ImVec2(bb_max.x, groove_y1), groove_col, groove_h * 0.5f);
  dl->AddRectFilled(ImVec2(bb_min.x, groove_y0), ImVec2(hx, groove_y1), fill_col, groove_h * 0.5f);
  drawSliderHandle(dl, ImRect(ImVec2(hx - SLIDER_LENGTH * 0.5f, cy - handle_h * 0.5f),
                              ImVec2(hx + SLIDER_LENGTH * 0.5f, cy + handle_h * 0.5f)));
  return changed;
}

ImFont *boldFont() { return g_bold_font; }
ImFont *monoFont() { return g_mono_font; }

void pushMonoFont(float size) {
  if (!g_mono_font) return;
  size > 0.0f ? ImGui::PushFont(g_mono_font, size) : ImGui::PushFont(g_mono_font);
}
void popMonoFont() { if (g_mono_font) ImGui::PopFont(); }
void pushBoldFont() { if (g_bold_font) ImGui::PushFont(g_bold_font); }
void popBoldFont() { if (g_bold_font) ImGui::PopFont(); }
void pushLargeFont() { if (g_large_font) ImGui::PushFont(g_large_font); }
void popLargeFont() { if (g_large_font) ImGui::PopFont(); }
