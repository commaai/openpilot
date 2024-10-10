#include "selfdrive/ui/raylib/cameraview.h"

#include "common/util.h"

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

CameraView::CameraView(const std::string &name, VisionStreamType type) {
  client = std::make_unique<VisionIpcClient>(name, type, false);
  shader = LoadShaderFromMemory(NULL, frame_fragment_shader);
}

CameraView::~CameraView() {
  if (textureY.id) UnloadTexture(textureY);
  if (textureUV.id) UnloadTexture(textureUV);
  if (shader.id) UnloadShader(shader);
}

void CameraView::draw(const Rectangle &rec) {
  if (!ensureConnection()) return;

  auto buffer = client->recv(nullptr, 20);
  frame = buffer ? buffer : frame;
  if (!frame) return;

  UpdateTexture(textureY, frame->y);
  UpdateTexture(textureUV, frame->uv);

  // Calculate scaling factors to maintain aspect ratio
  float scale = std::min((float)rec.width / frame->width, (float)rec.height / frame->height);
  float x_offset = rec.x + (rec.width - (frame->width * scale)) / 2;
  float y_offset = rec.y + (rec.height - (frame->height * scale)) / 2;
  Rectangle src_rect = {0, 0, (float)frame->width, (float)frame->height};
  Rectangle dst_rect = {x_offset, y_offset, frame->width * scale, frame->height * scale};

  BeginShaderMode(shader);
  SetShaderValueTexture(shader, GetShaderLocation(shader, "texture1"), textureUV);
  DrawTexturePro(textureY, src_rect, dst_rect, Vector2{0, 0}, 0.0, WHITE);
  EndShaderMode();
}

bool CameraView::ensureConnection() {
  if (!client->connected) {
    frame = nullptr;
    if (!client->connect(false)) return false;

    if (textureY.id) UnloadTexture(textureY);
    if (textureUV.id) UnloadTexture(textureUV);
    // Create textures for Y and UV planes
    const auto &buf = client->buffers[0];
    textureY = LoadTextureFromImage(Image{nullptr, (int)buf.stride, (int)buf.height, 1, PIXELFORMAT_UNCOMPRESSED_GRAYSCALE});
    textureUV = LoadTextureFromImage(Image{nullptr, (int)buf.stride / 2, (int)buf.height / 2, 1, PIXELFORMAT_UNCOMPRESSED_GRAY_ALPHA});
  }
  return true;
}
