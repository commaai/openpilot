#include "system/ui/raylib/util.h"

#include <array>
#include <filesystem>

#undef GREEN
#undef RED
#undef YELLOW
#include "common/swaglog.h"
#include "system/hardware/hw.h"

constexpr std::array<const char *, static_cast<int>(FontWeight::Count)> FONT_FILE_PATHS = {
    "../../selfdrive/assets/fonts/Inter-Black.ttf",
    "../../selfdrive/assets/fonts/Inter-Bold.ttf",
    "../../selfdrive/assets/fonts/Inter-ExtraBold.ttf",
    "../../selfdrive/assets/fonts/Inter-ExtraLight.ttf",
    "../../selfdrive/assets/fonts/Inter-Medium.ttf",
    "../../selfdrive/assets/fonts/Inter-Regular.ttf",
    "../../selfdrive/assets/fonts/Inter-SemiBold.ttf",
    "../../selfdrive/assets/fonts/Inter-Thin.ttf",
};

Texture2D LoadTextureResized(const char *fileName, int size) {
  Image img = LoadImage(fileName);
  ImageResize(&img, size, size);
  Texture2D texture = LoadTextureFromImage(img);
  return texture;
}

App *pApp = nullptr;

App::App(const char *title, int fps) {
  // Ensure the current dir matches the exectuable's directory
  auto self_path = util::readlink("/proc/self/exe");
  auto exe_dir = std::filesystem::path(self_path).parent_path();
  chdir(exe_dir.c_str());

  Hardware::set_display_power(true);
  Hardware::set_brightness(65);

  // SetTraceLogLevel(LOG_NONE);
  InitWindow(2160, 1080, title);
  SetTargetFPS(fps);

  // Load fonts
  fonts_.reserve(FONT_FILE_PATHS.size());
  for (int i = 0; i < FONT_FILE_PATHS.size(); ++i) {
    fonts_.push_back(LoadFontEx(FONT_FILE_PATHS[i], 120, nullptr, 250));
  }

  pApp = this;
}

App::~App() {
  for (auto &font : fonts_) {
    UnloadFont(font);
  }

  CloseWindow();
  pApp = nullptr;
}

const Font &App::getFont(FontWeight weight) const {
  return fonts_[static_cast<int>(weight)];
}
