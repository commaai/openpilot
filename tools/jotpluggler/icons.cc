#include "tools/jotpluggler/app.h"
#include "tools/jotpluggler/common.h"

#include <cmath>

void icon_add_font(float size, bool merge, const ImFont *base_font) {
  const std::filesystem::path ttf = repo_root() / "third_party" / "bootstrap" / "bootstrap-icons.ttf";
  ImGuiIO &io = ImGui::GetIO();
  ImFontConfig config;
  config.MergeMode = merge;
  config.GlyphMinAdvanceX = size;
  if (base_font != nullptr) {
    ImFontBaked *baked = const_cast<ImFont *>(base_font)->GetFontBaked(size);
    const float base_center = baked != nullptr ? (baked->Ascent + baked->Descent) * 0.5f : size * 0.5f;
    config.GlyphOffset.y = std::round(size * 0.5f - base_center);
  }
  static const ImWchar ranges[] = {0xF000, 0xF8FF, 0};
  io.Fonts->AddFontFromFileTTF(ttf.c_str(), size, &config, ranges);
}

bool icon_menu_item(const char *glyph, const char *label, const char *shortcut, bool selected, bool enabled) {
  assert(glyph != nullptr && glyph[0] != '\0');
  return ImGui::MenuItem(util::string_format("%s  %s", glyph, label).c_str(), shortcut, selected, enabled);
}
