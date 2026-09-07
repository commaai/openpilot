#include "tools/cabana/ui/theme.h"

#include <algorithm>
#include <cmath>
#include <filesystem>

#include "implot.h"
#include "tools/cabana/core/settings.h"

namespace fs = std::filesystem;

namespace {

constexpr Palette DARK_PALETTE = {
  .text = rgb(0xf8f9f9), .text_disabled = rgb(0xb8c0c4),
  .window = rgb(0x1d2225), .surface = rgb(0x30373b),
  .frame = rgb(0x1e2224), .frame_hovered = rgb(0x394044), .frame_active = rgb(0x424a4f),
  .button = rgb(0x424a4f), .button_hovered = rgb(0x535f64), .button_active = rgb(0x175886),
  .header = rgb(0x175886), .header_hovered = rgb(0x24455e), .header_active = rgb(0x1c6ea8),
  .accent = rgb(0x57a9e3),
  .border = rgb(0x65737a), .separator = rgb(0x4b5559),
  .tab = rgb(0x272c2f), .tab_hovered = rgb(0x424a4f), .table_header = rgb(0x424a4f),
  .grid = rgb(0x65737a, 0.45f), .badge = rgb(0x808080),
};

constexpr Palette LIGHT_PALETTE = {
  .text = rgb(0x1e2224), .text_disabled = rgb(0x535f64),
  .window = rgb(0xeeeff0), .surface = rgb(0xffffff),
  .frame = rgb(0xf8f9f9), .frame_hovered = rgb(0xeeeff0), .frame_active = rgb(0xddeef9),
  .button = rgb(0xe3e6e8), .button_hovered = rgb(0xd8dcdf), .button_active = rgb(0xbcddf4),
  .header = rgb(0xbcddf4), .header_hovered = rgb(0xddeef9), .header_active = rgb(0x9fcbec),
  .accent = rgb(0x1c6ea8),
  .border = rgb(0x98a3a9), .separator = rgb(0xcdd3d6),
  .tab = rgb(0xe3e6e8), .tab_hovered = rgb(0xddeef9), .table_header = rgb(0xd8dcdf),
  .grid = rgb(0x98a3a9, 0.4f), .badge = rgb(0xa0a0a4),
};

bool g_dark = false;
const Palette *g_palette = &LIGHT_PALETTE;
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

ImVec4 alpha(ImVec4 c, float a) { return ImVec4(c.x, c.y, c.z, a); }

}  // namespace

void loadFonts() {
  ImGuiIO &io = ImGui::GetIO();
  const fs::path fonts = fs::path(CABANA_FONTS_DIR);
  g_ui_font = addFont(fonts / "Inter-Regular.ttf", UI_FONT_SIZE);
  g_bold_font = addFont(fonts / "Inter-SemiBold.ttf", UI_FONT_SIZE);
  g_mono_font = addFont(fonts / "JetBrainsMono-Medium.ttf", 15.0f);
  g_large_font = addFont(fonts / "Inter-Bold.ttf", 50.0f);
  if (g_ui_font != nullptr) io.FontDefault = g_ui_font;
  if (g_bold_font == nullptr) g_bold_font = g_ui_font;
  if (g_mono_font == nullptr) g_mono_font = g_ui_font;
  if (g_large_font == nullptr) g_large_font = g_bold_font;
}

void applyTheme(int theme) {
  g_dark = theme == DARK_THEME;
  g_palette = g_dark ? &DARK_PALETTE : &LIGHT_PALETTE;
  const Palette &p = *g_palette;
  const ImVec4 none(0, 0, 0, 0);

  ImGuiStyle &style = ImGui::GetStyle();
  style = ImGuiStyle();
  style.WindowRounding = 6.0f;  // dialogs and tooltips; docked panels and os windows ignore it
  style.ChildRounding = 6.0f;
  style.PopupRounding = 6.0f;
  style.FrameRounding = 4.0f;
  style.GrabRounding = 4.0f;
  style.ScrollbarRounding = 4.0f;
  style.TabRounding = 4.0f;
  style.WindowBorderSize = 1.0f;
  style.FrameBorderSize = 1.0f;
  style.TabBorderSize = 1.0f;
  style.WindowPadding = ImVec2(12.0f, 10.0f);
  style.FramePadding = ImVec2(9.0f, 5.0f);
  style.ItemSpacing = ImVec2(10.0f, 8.0f);
  style.CellPadding = ImVec2(6.0f, 4.0f);
  style.ScrollbarSize = 14.0f;
  style.GrabMinSize = 13.0f;

  ImVec4 *c = style.Colors;
  c[ImGuiCol_Text] = p.text;
  c[ImGuiCol_TextDisabled] = p.text_disabled;
  c[ImGuiCol_WindowBg] = c[ImGuiCol_ScrollbarBg] = c[ImGuiCol_DockingEmptyBg] = p.window;
  c[ImGuiCol_MenuBarBg] = p.surface;
  c[ImGuiCol_TitleBg] = c[ImGuiCol_TitleBgActive] = c[ImGuiCol_TitleBgCollapsed] = p.window;
  c[ImGuiCol_ChildBg] = c[ImGuiCol_PopupBg] = p.surface;
  c[ImGuiCol_Border] = c[ImGuiCol_TableBorderStrong] = p.border;
  c[ImGuiCol_Separator] = c[ImGuiCol_TableBorderLight] = p.separator;
  c[ImGuiCol_BorderShadow] = c[ImGuiCol_ResizeGrip] = c[ImGuiCol_TableRowBg] = none;
  c[ImGuiCol_FrameBg] = p.frame;
  c[ImGuiCol_FrameBgHovered] = p.frame_hovered;
  c[ImGuiCol_FrameBgActive] = p.frame_active;
  c[ImGuiCol_Button] = p.button;
  c[ImGuiCol_ButtonHovered] = c[ImGuiCol_SliderGrab] = p.button_hovered;
  c[ImGuiCol_ScrollbarGrab] = p.border;
  c[ImGuiCol_ScrollbarGrabHovered] = p.text_disabled;
  c[ImGuiCol_ButtonActive] = p.button_active;
  c[ImGuiCol_Header] = p.header;
  c[ImGuiCol_HeaderHovered] = p.header_hovered;
  c[ImGuiCol_HeaderActive] = p.header_active;
  c[ImGuiCol_CheckMark] = c[ImGuiCol_NavCursor] = c[ImGuiCol_SliderGrabActive] = c[ImGuiCol_ScrollbarGrabActive] = p.accent;
  c[ImGuiCol_SeparatorActive] = c[ImGuiCol_ResizeGripActive] = c[ImGuiCol_DragDropTarget] = p.accent;
  c[ImGuiCol_TabSelectedOverline] = c[ImGuiCol_PlotHistogram] = p.accent;
  c[ImGuiCol_SeparatorHovered] = c[ImGuiCol_ResizeGripHovered] = alpha(p.accent, 0.6f);
  c[ImGuiCol_DockingPreview] = alpha(p.accent, 0.5f);
  c[ImGuiCol_TextSelectedBg] = alpha(p.accent, 0.35f);
  c[ImGuiCol_Tab] = c[ImGuiCol_TabDimmed] = p.tab;
  c[ImGuiCol_TabHovered] = p.tab_hovered;
  c[ImGuiCol_TabSelected] = c[ImGuiCol_TabDimmedSelected] = p.surface;
  c[ImGuiCol_TabDimmedSelectedOverline] = none;
  c[ImGuiCol_TableHeaderBg] = p.table_header;
  c[ImGuiCol_TableRowBgAlt] = g_dark ? ImVec4(1, 1, 1, 0.065f) : ImVec4(0, 0, 0, 0.045f);
  c[ImGuiCol_PlotLines] = p.text;
  // ImGuiStyle() seeds every slot from the dark theme: set the rest so the light theme does not keep a white caret.
  c[ImGuiCol_InputTextCursor] = c[ImGuiCol_UnsavedMarker] = p.text;
  c[ImGuiCol_TextLink] = c[ImGuiCol_PlotLinesHovered] = c[ImGuiCol_PlotHistogramHovered] = c[ImGuiCol_NavWindowingHighlight] = p.accent;
  c[ImGuiCol_TreeLines] = p.separator;
  c[ImGuiCol_DragDropTargetBg] = alpha(p.accent, 0.2f);
  // Disable the modal dim fade to make dialogs appear immediately.
  c[ImGuiCol_ModalWindowDimBg] = c[ImGuiCol_NavWindowingDimBg] = none;

  // ChartView::drawAxes() pushes the other plot colors it needs; the rest are auto colors derived from the ImGui style.
  ImPlot::GetStyle().Colors[ImPlotCol_AxisGrid] = p.grid;
}

bool isDarkTheme() { return g_dark; }
const Palette &palette() { return *g_palette; }

CabanaColor signalFillColor(const CabanaColor &c) {
  if (!g_dark) return c;
  auto [h, s, v] = c.hsv();
  return CabanaColor::fromHsv(h, std::min(1.0f, s * 1.4f), v * 0.8f, c.a / 255.0f);
}

ImFont *boldFont() { return g_bold_font; }

void pushMonoFont(float size) {
  if (!g_mono_font) return;
  size > 0.0f ? ImGui::PushFont(g_mono_font, size) : ImGui::PushFont(g_mono_font);
}
void popMonoFont() { if (g_mono_font) ImGui::PopFont(); }
void pushBoldFont() { if (g_bold_font) ImGui::PushFont(g_bold_font); }
void popBoldFont() { if (g_bold_font) ImGui::PopFont(); }
void pushLargeFont() { if (g_large_font) ImGui::PushFont(g_large_font); }
void popLargeFont() { if (g_large_font) ImGui::PopFont(); }
