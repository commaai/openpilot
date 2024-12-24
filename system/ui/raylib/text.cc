#include <string>
#include <vector>
#include <cstring>

#include "system/hardware/hw.h"
#include "system/ui/raylib/util.h"

constexpr int kMargin = 50;
constexpr int kFontSize = 70;
constexpr int kButtonPadding = 50;
constexpr int kButtonSidePadding = 100;
constexpr int kButtonRadius = 20;
constexpr int kTextSpacing = 0.0f;

struct Button {
  Rectangle bounds{0, 0, 0, 0};
  const char* text{nullptr};
  bool hovered{false};

  void draw() {
    Color color = hovered ? RAYLIB_GRAY : RAYLIB_BLACK;
    DrawRectangleRounded(bounds, 0.2f, kButtonRadius, color);
    DrawRectangleRoundedLines(bounds, 0.2f, kButtonRadius, RAYLIB_WHITE);

    Vector2 textSize = MeasureTextEx(getFont(), text, kFontSize, kTextSpacing);
    Vector2 textPos = {
      bounds.x + (bounds.width - textSize.x) / 2,
      bounds.y + (bounds.height - textSize.y) / 2
    };
    DrawTextEx(getFont(), text, textPos, kFontSize, kTextSpacing, RAYLIB_WHITE);
  }

  bool checkHover(Vector2 mousePos) {
    hovered = CheckCollisionPointRec(mousePos, bounds);
    return hovered;
  }
};

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printf("Usage: ./text <message>\n");
    return 1;
  }

  SetConfigFlags(FLAG_MSAA_4X_HINT);

  initApp("text", 60);
  SetExitKey(0);

  const char* displayText = argv[1];
  if (!displayText || strlen(displayText) == 0) {
    printf("Error: Empty message\n");
    return 1;
  }

  Button button;
#ifdef __aarch64__
  button.text = "Reboot";
#else
  button.text = "Exit";
#endif

  while (!WindowShouldClose()) {
    if (IsKeyPressed(KEY_ESCAPE)) break;

    BeginDrawing();
    ClearBackground(RAYLIB_BLACK);

    // Update button position and size
    float buttonWidth = MeasureTextEx(getFont(), button.text, kFontSize, kTextSpacing).x + 2 * kButtonSidePadding;
    button.bounds = {
      static_cast<float>(GetScreenWidth() - buttonWidth - kMargin * 2),
      static_cast<float>(GetScreenHeight() - kFontSize - 2 * kButtonPadding - kMargin),
      buttonWidth,
      static_cast<float>(kFontSize + 2 * kButtonPadding)
    };

    // Handle mouse input
    Vector2 mousePos = GetMousePosition();
    button.checkHover(mousePos);

    if (button.hovered && IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
#ifdef __aarch64__
      Hardware::reboot();
#else
      CloseWindow();
      return 0;
#endif
    }

    // Draw text in container
    Rectangle textBox = {
      static_cast<float>(kMargin),
      static_cast<float>(kMargin),
      static_cast<float>(GetScreenWidth() - 2 * kMargin),
      static_cast<float>(GetScreenHeight() - button.bounds.height - 3 * kMargin)
    };

    DrawTextBoxed(displayText, textBox, kFontSize, kTextSpacing, true, RAYLIB_WHITE);

    button.draw();

    EndDrawing();
  }

  CloseWindow();
  return 0;
}
