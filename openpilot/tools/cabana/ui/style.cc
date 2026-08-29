#include "tools/cabana/ui/app.h"

#include <cmath>
#include <filesystem>

#include "implot.h"
#include "tools/cabana/core/settings.h"
#include "tools/cabana/ui/imgui_util.h"
#include "tools/cabana/utils/util.h"

#ifdef __APPLE__
#include <CoreFoundation/CoreFoundation.h>
#endif

namespace fs = std::filesystem;

namespace {
// Qt's "Automatic" theme is QStyle::standardPalette(), which follows the system appearance on macOS
// and stays light elsewhere. Match that so both frontends pick the same theme.
bool systemPrefersDark() {
#ifdef __APPLE__
  bool dark = false;
  if (CFPropertyListRef v = CFPreferencesCopyAppValue(CFSTR("AppleInterfaceStyle"), kCFPreferencesAnyApplication)) {
    dark = CFGetTypeID(v) == CFStringGetTypeID() &&
           CFStringCompare((CFStringRef)v, CFSTR("Dark"), kCFCompareCaseInsensitive) == kCFCompareEqualTo;
    CFRelease(v);
  }
  return dark;
#else
  return false;
#endif
}

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
  const bool dark = theme == DARK_THEME || (theme == AUTO_THEME && systemPrefersDark());
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
  style.TabRounding = 0.0f;
  style.WindowBorderSize = 1.0f;
  style.FrameBorderSize = 1.0f;
  style.WindowPadding = ImVec2(8.0f, 7.0f);
  style.FramePadding = ImVec2(6.0f, 3.0f);
  style.ItemSpacing = ImVec2(8.0f, 5.0f);

  if (dark) {
    // the whole palette comes from DarkTheme, so no imgui default dark colors bleed through
    auto c = [](const CabanaColor &col) { return colorRgb(col.r, col.g, col.b); };
    auto lighter = [&c](const CabanaColor &col, int f) { return c(col.lighter(f)); };
    style.Colors[ImGuiCol_WindowBg] = c(DarkTheme::window);
    style.Colors[ImGuiCol_ChildBg] = c(DarkTheme::base);
    style.Colors[ImGuiCol_PopupBg] = c(DarkTheme::base);
    style.Colors[ImGuiCol_Text] = c(DarkTheme::text);
    style.Colors[ImGuiCol_TextDisabled] = c(DarkTheme::disabled_text);
    style.Colors[ImGuiCol_Border] = c(DarkTheme::light);
    style.Colors[ImGuiCol_BorderShadow] = ImVec4(0, 0, 0, 0);

    style.Colors[ImGuiCol_FrameBg] = c(DarkTheme::base);
    style.Colors[ImGuiCol_FrameBgHovered] = lighter(DarkTheme::base, 115);
    style.Colors[ImGuiCol_FrameBgActive] = lighter(DarkTheme::base, 130);

    style.Colors[ImGuiCol_Button] = c(DarkTheme::button);
    style.Colors[ImGuiCol_ButtonHovered] = lighter(DarkTheme::button, 115);
    style.Colors[ImGuiCol_ButtonActive] = lighter(DarkTheme::button, 130);
    style.Colors[ImGuiCol_CheckMark] = c(DarkTheme::bright_text);
    style.Colors[ImGuiCol_SliderGrab] = c(DarkTheme::highlight);
    style.Colors[ImGuiCol_SliderGrabActive] = lighter(DarkTheme::highlight, 115);

    style.Colors[ImGuiCol_Header] = c(DarkTheme::highlight);
    style.Colors[ImGuiCol_HeaderHovered] = lighter(DarkTheme::highlight, 115);
    style.Colors[ImGuiCol_HeaderActive] = lighter(DarkTheme::highlight, 130);

    style.Colors[ImGuiCol_MenuBarBg] = c(DarkTheme::window);
    style.Colors[ImGuiCol_TitleBg] = c(DarkTheme::window);
    style.Colors[ImGuiCol_TitleBgActive] = c(DarkTheme::window);
    style.Colors[ImGuiCol_TitleBgCollapsed] = c(DarkTheme::window);

    style.Colors[ImGuiCol_ScrollbarBg] = c(DarkTheme::window);
    style.Colors[ImGuiCol_ScrollbarGrab] = c(DarkTheme::light);
    style.Colors[ImGuiCol_ScrollbarGrabHovered] = lighter(DarkTheme::light, 115);
    style.Colors[ImGuiCol_ScrollbarGrabActive] = lighter(DarkTheme::light, 130);

    style.Colors[ImGuiCol_Separator] = c(DarkTheme::light);
    style.Colors[ImGuiCol_SeparatorHovered] = c(DarkTheme::highlight);
    style.Colors[ImGuiCol_SeparatorActive] = c(DarkTheme::highlight);
    style.Colors[ImGuiCol_ResizeGrip] = c(DarkTheme::light);
    style.Colors[ImGuiCol_ResizeGripHovered] = c(DarkTheme::highlight);
    style.Colors[ImGuiCol_ResizeGripActive] = c(DarkTheme::highlight);

    style.Colors[ImGuiCol_Tab] = c(DarkTheme::window);
    style.Colors[ImGuiCol_TabHovered] = lighter(DarkTheme::base, 115);
    style.Colors[ImGuiCol_TabSelected] = c(DarkTheme::base);
    style.Colors[ImGuiCol_TabDimmed] = c(DarkTheme::window);
    style.Colors[ImGuiCol_TabDimmedSelected] = c(DarkTheme::base);
    style.Colors[ImGuiCol_DockingEmptyBg] = c(DarkTheme::window);
    style.Colors[ImGuiCol_DockingPreview] = c(DarkTheme::highlight);

    style.Colors[ImGuiCol_TableHeaderBg] = c(DarkTheme::window);
    style.Colors[ImGuiCol_TableBorderStrong] = c(DarkTheme::light);
    style.Colors[ImGuiCol_TableBorderLight] = c(DarkTheme::dark);
    style.Colors[ImGuiCol_TableRowBg] = ImVec4(0, 0, 0, 0);
    style.Colors[ImGuiCol_TableRowBgAlt] = c(DarkTheme::base);
    style.Colors[ImGuiCol_TextSelectedBg] = c(DarkTheme::highlight);
  } else {
    style.Colors[ImGuiCol_WindowBg] = colorRgb(250, 250, 251);
    style.Colors[ImGuiCol_ChildBg] = colorRgb(255, 255, 255);
    style.Colors[ImGuiCol_MenuBarBg] = colorRgb(232, 236, 241);
    style.Colors[ImGuiCol_Text] = colorRgb(40, 44, 50);
    style.Colors[ImGuiCol_Border] = colorRgb(194, 198, 204);
    style.Colors[ImGuiCol_Tab] = colorRgb(219, 224, 230);
    style.Colors[ImGuiCol_TabSelected] = colorRgb(250, 251, 253);
    style.Colors[ImGuiCol_DockingEmptyBg] = colorRgb(244, 246, 248);
    // QPalette::Highlight is opaque in Qt; the imgui light defaults are translucent
    const CabanaColor highlight{48, 140, 198};
    style.Colors[ImGuiCol_Header] = colorRgb(highlight.r, highlight.g, highlight.b);
    style.Colors[ImGuiCol_HeaderHovered] = colorRgb(highlight.lighter(115).r, highlight.lighter(115).g, highlight.lighter(115).b);
    style.Colors[ImGuiCol_HeaderActive] = colorRgb(highlight.lighter(130).r, highlight.lighter(130).g, highlight.lighter(130).b);
    style.Colors[ImGuiCol_TextSelectedBg] = colorRgb(highlight.r, highlight.g, highlight.b);
  }
}

bool isDarkTheme() { return g_dark; }

ImU32 highlightedTextColor() {
  return g_dark ? IM_COL32(DarkTheme::window_text.r, DarkTheme::window_text.g, DarkTheme::window_text.b, 255)
                : IM_COL32(255, 255, 255, 255);
}

void pushMonoFont() { if (g_mono_font) ImGui::PushFont(g_mono_font); }
void popMonoFont() { if (g_mono_font) ImGui::PopFont(); }
void pushBoldFont() { if (g_bold_font) ImGui::PushFont(g_bold_font); }
void popBoldFont() { if (g_bold_font) ImGui::PopFont(); }
void pushLargeFont() { if (g_large_font) ImGui::PushFont(g_large_font); }
void popLargeFont() { if (g_large_font) ImGui::PopFont(); }
