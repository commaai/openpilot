#include "tools/jotpluggler/jotpluggler.h"

#include <cstdio>

namespace {

ImFont *g_icon_font = nullptr;

const std::filesystem::path &font_path() {
  static const std::filesystem::path path = []() -> std::filesystem::path {
#ifdef JOTP_REPO_ROOT
    return std::filesystem::path(JOTP_REPO_ROOT) / "third_party" / "bootstrap" / "bootstrap-icons.ttf";
#else
    return std::filesystem::current_path() / "third_party" / "bootstrap" / "bootstrap-icons.ttf";
#endif
  }();
  return path;
}

}  // namespace

void icon_add_font(float size, bool merge) {
  const auto &ttf = font_path();
  ImGuiIO &io = ImGui::GetIO();
  ImFontConfig config;
  config.MergeMode = merge;
  config.GlyphMinAdvanceX = size;
  static const ImWchar ranges[] = {0xF000, 0xF8FF, 0};
  ImFont *font = io.Fonts->AddFontFromFileTTF(ttf.c_str(), size, &config, ranges);
  if (!merge && font != nullptr) {
    g_icon_font = font;
  }
}

ImFont *icon_font() {
  return g_icon_font;
}

bool icon_menu_item(const char *glyph,
                    const char *label,
                    const char *shortcut,
                    bool selected,
                    bool enabled) {
  char buf[256];
  if (glyph != nullptr && glyph[0] != '\0') {
    std::snprintf(buf, sizeof(buf), "%s  %s", glyph, label);
  } else {
    std::snprintf(buf, sizeof(buf), "   %s", label);
  }
  return ImGui::MenuItem(buf, shortcut, selected, enabled);
}
