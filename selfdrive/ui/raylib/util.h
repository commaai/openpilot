#pragma once

#include <string>

#include "third_party/raylib/include/raylib.h"

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

void initApp(const char *title, int fps);
const Font& getFont(FontWeight weight = FontWeight::Normal);
Texture2D LoadTextureResized(const char *fileName, int size);