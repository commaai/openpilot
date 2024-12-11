#include "system/ui/raylib/util.h"

#include <array>

#undef GREEN
#undef RED
#undef YELLOW
#include "common/swaglog.h"
#include "system/hardware/hw.h"

constexpr std::array<const char *, static_cast<int>(FontWeight::Count)> FONT_FILE_PATHS = {
    "../../assets/fonts/Inter-Black.ttf",
    "../../assets/fonts/Inter-Bold.ttf",
    "../../assets/fonts/Inter-ExtraBold.ttf",
    "../../assets/fonts/Inter-ExtraLight.ttf",
    "../../assets/fonts/Inter-Medium.ttf",
    "../../assets/fonts/Inter-Regular.ttf",
    "../../assets/fonts/Inter-SemiBold.ttf",
    "../../assets/fonts/Inter-Thin.ttf",
};

struct FontManager {
  FontManager() {
    for (int i = 0; i < fonts.size(); ++i) {
      fonts[i] = LoadFontEx(FONT_FILE_PATHS[i], 120, nullptr, 250);
    }
  }

  ~FontManager() {
    for (auto &f : fonts) UnloadFont(f);
  }

  std::array<Font, static_cast<int>(FontWeight::Count)> fonts;
};

const Font& getFont(FontWeight weight) {
  static FontManager font_manager;
  return font_manager.fonts[(int)weight];
}

Texture2D LoadTextureResized(const char *fileName, int size) {
  Image img = LoadImage(fileName);
  ImageResize(&img, size, size);
  Texture2D texture = LoadTextureFromImage(img);
  return texture;
}

void initApp(const char *title, int fps) {
  Hardware::set_display_power(true);
  Hardware::set_brightness(65);
  // SetTraceLogLevel(LOG_NONE);
  InitWindow(2160, 1080, title);
  SetTargetFPS(fps);
}
