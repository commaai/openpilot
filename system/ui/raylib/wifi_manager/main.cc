#include "system/ui/raylib/util.h"
#include "system/ui/raylib/wifi_manager/wifi_manager.h"
#include "third_party/raylib/include/raygui.h"

int main() {
  App app("Wi-Fi Manager", 30);

  WifiManager wifi_manager;
  while (!WindowShouldClose()) {
    BeginDrawing();
      ClearBackground(RAYLIB_BLACK);
      GuiLabel({40, 20, 300, 40}, "Wi-Fi Manager");
      wifi_manager.render({40, 100, GetScreenWidth() - 80.0f, GetScreenHeight() - 180.0f});
    EndDrawing();
  }
  return 0;
}
