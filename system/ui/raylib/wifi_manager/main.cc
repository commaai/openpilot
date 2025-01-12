#include "system/ui/raylib/util.h"
#include "system/ui/raylib/wifi_manager/wifi_manager.h"

int main() {
  initApp("Wi-Fi Manager", 30);

  WifiManager wifi_manager;
  while (!WindowShouldClose()) {
    BeginDrawing();
      ClearBackground(RAYLIB_BLACK);
      wifi_manager.draw({0, 0, (float)GetScreenWidth(), (float)GetScreenHeight()});
    EndDrawing();
  }

  CloseWindow();
  return 0;
}
