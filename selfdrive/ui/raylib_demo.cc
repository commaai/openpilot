#include <raylib.h>
#include "common/util.h"
#include "msgq/visionipc/visionipc_client.h"

const char frame_fragment_shader[] = R"(
  #version 330 core
  in vec2 fragTexCoord;
  uniform sampler2D texture0;  // Y plane
  uniform sampler2D texture1;  // UV plane
  out vec4 fragColor;
  void main() {
    float y = texture(texture0, fragTexCoord).r;
    vec2 uv = texture(texture1, fragTexCoord).ra - 0.5;
    float r = y + 1.402 * uv.y;
    float g = y - 0.344 * uv.x - 0.714 * uv.y;
    float b = y + 1.772 * uv.x;
    fragColor = vec4(r, g, b, 1.0);
  }
)";

int main(void) {
  VisionIpcClient client("camerad", VISION_STREAM_ROAD, false);
  while (!client.connect()) {
    util::sleep_for(100);
  }

  InitWindow(1024, 768, "Raylib CameraView Demo");

  Shader shader = LoadShaderFromMemory(NULL, frame_fragment_shader);

  // Get stream dimensions
  int stream_width = client.buffers[0].width;
  int stream_height = client.buffers[0].height;
  int stream_stride = client.buffers[0].stride;

  // Create textures for Y and UV planes
  Texture2D textureY = LoadTextureFromImage({nullptr, stream_stride, stream_height, 1, PIXELFORMAT_UNCOMPRESSED_GRAYSCALE});
  Texture2D textureUV = LoadTextureFromImage({nullptr, stream_stride / 2, stream_height / 2, 1, PIXELFORMAT_UNCOMPRESSED_GRAY_ALPHA});

  // Calculate scaling factors to maintain aspect ratio
  float scale = std::min((float)GetScreenWidth() / stream_width, (float)GetScreenHeight() / stream_height);
  float x_offset = (GetScreenWidth() - (stream_width * scale)) / 2;
  float y_offset = (GetScreenHeight() - (stream_height * scale)) / 2;
  Rectangle src_rect = {0, 0, (float)stream_width, (float)stream_height};
  Rectangle dst_rect = {x_offset, y_offset, stream_width * scale, stream_height * scale};

  while (!WindowShouldClose()) {
    auto buf = client.recv();
    if (!buf) {
      util::sleep_for(100);
      continue;
    }

    UpdateTexture(textureY, buf->y);
    UpdateTexture(textureUV, buf->uv);

    BeginDrawing();
      ClearBackground(BLACK);
        BeginShaderMode(shader);
          SetShaderValueTexture(shader, GetShaderLocation(shader, "texture1"), textureUV);
          DrawTexturePro(textureY, src_rect, dst_rect, Vector2{0, 0}, 0.0, WHITE);
        EndShaderMode();
      DrawText("RAYLIB CAMERAVIEW", 10, 10, 20, WHITE);
    EndDrawing();
  }

  UnloadTexture(textureY);
  UnloadTexture(textureUV);
  UnloadShader(shader);
  CloseWindow();

  return 0;
}
