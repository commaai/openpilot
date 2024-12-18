#include <algorithm>
#include <cmath>
#include <iostream>

#include "system/ui/raylib/util.h"

constexpr int kProgressBarWidth = 1000;
constexpr int kProgressBarHeight = 20;
constexpr float kRotationRate = 12.0f;
constexpr int kMargin = 200;
constexpr int kTextureSize = 360;
constexpr int kFontSize = 80;

int main(int argc, char *argv[]) {
  initApp("spinner", 30);

  // Turn off input buffering for std::cin
  std::cin.sync_with_stdio(false);
  std::cin.tie(nullptr);

  Texture2D commaTexture = LoadTextureResized("../../selfdrive/assets/img_spinner_comma.png", kTextureSize);
  Texture2D spinnerTexture = LoadTextureResized("../../selfdrive/assets/img_spinner_track.png", kTextureSize);

  float rotation = 0.0f;
  std::string userInput;

  while (!WindowShouldClose()) {
    BeginDrawing();
    ClearBackground(RAYLIB_BLACK);

    rotation = fmod(rotation + kRotationRate, 360.0f);
    Vector2 center = {GetScreenWidth() / 2.0f, GetScreenHeight() / 2.0f};
    const Vector2 spinnerOrigin{kTextureSize / 2.0f, kTextureSize / 2.0f};
    const Vector2 commaPosition{center.x - kTextureSize / 2.0f, center.y - kTextureSize / 2.0f};

    // Draw rotating spinner and static comma logo
    DrawTexturePro(spinnerTexture, {0, 0, (float)kTextureSize, (float)kTextureSize},
                   {center.x, center.y, (float)kTextureSize, (float)kTextureSize},
                   spinnerOrigin, rotation, RAYLIB_WHITE);
    DrawTextureV(commaTexture, commaPosition, RAYLIB_WHITE);

    // Check for user input
    if (std::cin.rdbuf()->in_avail() > 0) {
      std::getline(std::cin, userInput);
    }

    // Display either a progress bar or user input text based on input
    if (!userInput.empty()) {
      float yPos = GetScreenHeight() - kMargin - kProgressBarHeight;
      if (std::all_of(userInput.begin(), userInput.end(), ::isdigit)) {
        Rectangle bar = {center.x - kProgressBarWidth / 2.0f, yPos, kProgressBarWidth, kProgressBarHeight};
        DrawRectangleRounded(bar, 0.5f, 10, RAYLIB_GRAY);

        int progress = std::clamp(std::stoi(userInput), 0, 100);
        bar.width *= progress / 100.0f;
        DrawRectangleRounded(bar, 0.5f, 10, RAYLIB_RAYWHITE);
      } else {
        Vector2 textSize = MeasureTextEx(getFont(), userInput.c_str(), kFontSize, 1.0);
        DrawTextEx(getFont(), userInput.c_str(), {center.x - textSize.x / 2, yPos}, kFontSize, 1.0, RAYLIB_WHITE);
      }
    }

    EndDrawing();
  }

  CloseWindow();
  return 0;
}
