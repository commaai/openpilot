#include "tools/cabana/ui/app.h"

#include <cmath>
#include <filesystem>

#include "implot.h"
#include "tools/cabana/core/settings.h"
#include "tools/cabana/ui/imgui_util.h"
#include "tools/cabana/utils/util.h"

namespace fs = std::filesystem;

namespace {
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
    auto c = [](const CabanaColor &col) { return colorRgb(col.r, col.g, col.b); };
    style.Colors[ImGuiCol_WindowBg] = c(DarkTheme::window);
    style.Colors[ImGuiCol_ChildBg] = c(DarkTheme::base);
    style.Colors[ImGuiCol_FrameBg] = c(DarkTheme::base);
    style.Colors[ImGuiCol_PopupBg] = c(DarkTheme::base);
    style.Colors[ImGuiCol_Text] = c(DarkTheme::text);
    style.Colors[ImGuiCol_TextDisabled] = c(DarkTheme::disabled_text);
    style.Colors[ImGuiCol_Button] = c(DarkTheme::button);
    style.Colors[ImGuiCol_Header] = c(DarkTheme::highlight);
    style.Colors[ImGuiCol_MenuBarBg] = c(DarkTheme::window);
    style.Colors[ImGuiCol_Border] = c(DarkTheme::light);
  } else {
    style.Colors[ImGuiCol_WindowBg] = colorRgb(250, 250, 251);
    style.Colors[ImGuiCol_ChildBg] = colorRgb(255, 255, 255);
    style.Colors[ImGuiCol_MenuBarBg] = colorRgb(232, 236, 241);
    style.Colors[ImGuiCol_Text] = colorRgb(40, 44, 50);
    style.Colors[ImGuiCol_Border] = colorRgb(194, 198, 204);
    style.Colors[ImGuiCol_Tab] = colorRgb(219, 224, 230);
    style.Colors[ImGuiCol_TabSelected] = colorRgb(250, 251, 253);
    style.Colors[ImGuiCol_DockingEmptyBg] = colorRgb(244, 246, 248);
  }
}

void pushMonoFont() { if (g_mono_font) ImGui::PushFont(g_mono_font); }
void popMonoFont() { if (g_mono_font) ImGui::PopFont(); }
void pushBoldFont() { if (g_bold_font) ImGui::PushFont(g_bold_font); }
void popBoldFont() { if (g_bold_font) ImGui::PopFont(); }
void pushLargeFont() { if (g_large_font) ImGui::PushFont(g_large_font); }
void popLargeFont() { if (g_large_font) ImGui::PopFont(); }
