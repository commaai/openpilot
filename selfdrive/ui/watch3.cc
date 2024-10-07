#include "selfdrive/ui/raylib/cameraview.h"

void renderCameraViews() {
  CameraView roadCamera("camerad", VISION_STREAM_ROAD);
  CameraView wideRoadCamera("camerad", VISION_STREAM_WIDE_ROAD);
  CameraView driverCamera("camerad", VISION_STREAM_DRIVER);

  while (!WindowShouldClose()) {
    float w = GetScreenWidth(), h = GetScreenHeight();
    Rectangle roadCameraRec = {0, 0, w, h / 2};
    Rectangle wideRoadCameraRec = {0, h / 2, w / 2, h / 2};
    Rectangle driverCameraRec = {w / 2, h / 2, w / 2, h / 2};

    BeginDrawing();
      ClearBackground(BLACK);
      roadCamera.draw(roadCameraRec);
      wideRoadCamera.draw(wideRoadCameraRec);
      driverCamera.draw(driverCameraRec);
    EndDrawing();
  }
}

int main(int argc, char *argv[]) {
  SetTraceLogLevel(LOG_NONE);
  SetConfigFlags(FLAG_WINDOW_RESIZABLE);
  InitWindow(0, 0, "Watch 3 Cameras");
  SetTargetFPS(20);

  renderCameraViews();

  CloseWindow();
  return 0;
}
