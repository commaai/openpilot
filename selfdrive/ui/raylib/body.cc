#include <algorithm>
#include <cmath>
#include <iostream>

#include "selfdrive/ui/raylib/util.h"
#include "selfdrive/ui/raylib/body.h"
#include "third_party/raylib/include/raylib.h"

constexpr int kProgressBarWidth = 1000;
constexpr int kProgressBarHeight = 20;
constexpr float kRotationRate = 12.0f;
constexpr int kMargin = 200;
constexpr int kTextureSize = 360;
constexpr int kFontSize = 80;

int main(int argc, char *argv[]) {
  initApp("body", 30);

  Texture2D awakeTexture = LoadTextureResized("../assets/body/awake.gif", kTextureSize);
  Texture2D sleepTexture = LoadTextureResized("../assets/body/sleep.gif", kTextureSize);

  float rotation = 0.0f;

  while (!WindowShouldClose()) {
    BeginDrawing();
    ClearBackground(BLACK);

    rotation = fmod(rotation + kRotationRate, 360.0f);
    Vector2 center = {GetScreenWidth() / 2.0f, GetScreenHeight() / 2.0f};
    const Vector2 spinnerOrigin{kTextureSize / 2.0f, kTextureSize / 2.0f};
    const Vector2 commaPosition{center.x - kTextureSize / 2.0f, center.y - kTextureSize / 2.0f};

    // Draw rotating spinner and static comma logo
    DrawTexturePro(awakeTexture, {0, 0, (float)kTextureSize, (float)kTextureSize},
                   {center.x, center.y, (float)kTextureSize, (float)kTextureSize},
                   spinnerOrigin, rotation, WHITE);
    DrawTextureV(sleepTexture, commaPosition, WHITE);


    EndDrawing();
  }

  CloseWindow();
  return 0;
}