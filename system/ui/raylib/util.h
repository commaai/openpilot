#pragma once

#include <string>

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

void initApp(const char *title, int fps);
const Font& getFont(FontWeight weight = FontWeight::Regular);
Texture2D LoadTextureResized(const char *fileName, int size);
void DrawTextBoxed(const char *text, Rectangle rec, float fontSize, float spacing, bool wordWrap, Color tint);
