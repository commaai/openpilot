#pragma once

#include <filesystem>
#include <string>
#include <string_view>

#include "imgui.h"

namespace loggy {

enum class LoggyThemeKind {
  Darcula,
  Light,
};

const std::filesystem::path &repo_root();
void load_fonts();
LoggyThemeKind loggy_theme_from_name(std::string_view name);
const char *loggy_theme_name(LoggyThemeKind theme);
const char *loggy_theme_label(LoggyThemeKind theme);
void apply_theme(LoggyThemeKind theme = LoggyThemeKind::Darcula);
ImVec4 clear_color();
ImVec4 color_rgb(int r, int g, int b, float alpha = 1.0f);
void push_bold_font();
void pop_bold_font();
void push_mono_font();
void pop_mono_font();
bool input_text_with_hint(const char *label, const char *hint, std::string *text, ImGuiInputTextFlags flags = 0);

}  // namespace loggy
