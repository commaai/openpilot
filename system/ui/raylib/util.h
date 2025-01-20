#pragma once

#include <string>
#include <vector>

#include "system/ui/raylib/raylib.h"

enum class FontWeight {
  Normal,
  Bold,
  ExtraBold,
  ExtraLight,
  Medium,
  Regular,
  SemiBold,
  Thin,
  Count // To represent the total number of fonts
};

Texture2D LoadTextureResized(const char *fileName, int size);

class App {
public:
  App(const char *title, int fps);
  ~App();
  const Font &getFont(FontWeight weight = FontWeight::Normal) const;

protected:
  std::vector<Font> fonts_;
};

// Global pointer to the App instance
extern App *pApp;
