#pragma once

#include <filesystem>

#include "imgui.h"

namespace loggy {

const std::filesystem::path &repo_root();
void load_fonts();
void apply_theme();
ImVec4 clear_color();
ImVec4 color_rgb(int r, int g, int b, float alpha = 1.0f);
void push_bold_font();
void pop_bold_font();
void push_mono_font();
void pop_mono_font();

}  // namespace loggy
