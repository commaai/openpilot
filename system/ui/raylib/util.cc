#include "system/ui/raylib/util.h"

#include <array>

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

  for (int i = 0; i < static_cast<int>(FontWeight::Count); i++) {
    SetTextureFilter(getFont(static_cast<FontWeight>(i)).texture, TEXTURE_FILTER_BILINEAR);
  }

  SetTargetFPS(fps);
}

void DrawTextBoxed(const char *text, Rectangle rec, float fontSize, float spacing, bool wordWrap, Color tint) {
  int length = TextLength(text);
  float textOffsetY = 0;
  float textOffsetX = 0.0f;
  float scaleFactor = fontSize/(float)getFont().baseSize;

  int state = wordWrap? 0 : 1;  // 0-Measure, 1-Draw
  int startLine = -1;
  int endLine = -1;
  int lastk = -1;

  for (int i = 0, k = 0; i < length; i++, k++) {
    int codepointByteCount = 0;
    int codepoint = GetCodepoint(&text[i], &codepointByteCount);
    int index = GetGlyphIndex(getFont(), codepoint);

    if (codepoint == 0x3f) codepointByteCount = 1;
    i += (codepointByteCount - 1);

    float glyphWidth = 0;
    if (codepoint != '\n') {
      glyphWidth = (getFont().glyphs[index].advanceX == 0) ?
        getFont().recs[index].width*scaleFactor :
        getFont().glyphs[index].advanceX*scaleFactor;

      if (i + 1 < length) glyphWidth = glyphWidth + spacing;
    }

    if (state == 0) {
      if ((codepoint == ' ') || (codepoint == '\t') || (codepoint == '\n')) endLine = i;

      if ((textOffsetX + glyphWidth) > rec.width) {
        endLine = (endLine < 1)? i : endLine;
        if (i == endLine) endLine -= codepointByteCount;
        if ((startLine + codepointByteCount) == endLine) endLine = (i - codepointByteCount);

        state = !state;
      } else if ((i + 1) == length) {
        endLine = i;
        state = !state;
      } else if (codepoint == '\n') state = !state;

      if (state == 1) {
        textOffsetX = 0;
        i = startLine;
        glyphWidth = 0;

        int tmp = lastk;
        lastk = k - 1;
        k = tmp;
      }
    } else {
      if (codepoint == '\n') {
        if (!wordWrap) {
          textOffsetY += (getFont().baseSize + getFont().baseSize/2)*scaleFactor;
          textOffsetX = 0;
        }
      } else {
        if (!wordWrap && ((textOffsetX + glyphWidth) > rec.width)) {
          textOffsetY += (getFont().baseSize + getFont().baseSize/2)*scaleFactor;
          textOffsetX = 0;
        }

        if ((textOffsetY + getFont().baseSize*scaleFactor) > rec.height) break;

        if ((codepoint != ' ') && (codepoint != '\t')) {
          DrawTextCodepoint(getFont(), codepoint,
            (Vector2){ rec.x + textOffsetX, rec.y + textOffsetY },
            fontSize, tint);
        }
      }

      if (wordWrap && (i == endLine)) {
        textOffsetY += (getFont().baseSize + getFont().baseSize/2)*scaleFactor;
        textOffsetX = 0;
        startLine = endLine;
        endLine = -1;
        glyphWidth = 0;
        k = lastk;

        state = !state;
      }
    }

    if ((textOffsetX != 0) || (codepoint != ' ')) textOffsetX += glyphWidth;
  }
}
