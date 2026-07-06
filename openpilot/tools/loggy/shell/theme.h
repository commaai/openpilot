#pragma once

#include <string>
#include <string_view>

#include "imgui.h"

namespace loggy {

enum class LoggyThemeKind {
  Darcula,
  Light,
};

void load_fonts();
LoggyThemeKind loggy_theme_from_name(std::string_view name);
const char *loggy_theme_name(LoggyThemeKind theme);
const char *loggy_theme_label(LoggyThemeKind theme);
void apply_theme(LoggyThemeKind theme = LoggyThemeKind::Darcula);
ImVec4 clear_color();
ImVec4 color_rgb(int r, int g, int b, float alpha = 1.0f);
ImVec4 plot_area_background_color();    // Plot pane hero chart fill, theme-resolved.
ImVec4 binary_grid_background_color();  // Binary pane idle bit-cell fill, theme-resolved.
void push_bold_font();
void pop_bold_font();
void push_mono_font();
void pop_mono_font();
bool input_text_with_hint(const char *label, const char *hint, std::string *text, ImGuiInputTextFlags flags = 0);
bool input_text_multiline(const char *label, std::string *text, ImVec2 size);

}  // namespace loggy
