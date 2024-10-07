#pragma once

#include <memory>
#include "msgq/visionipc/visionipc_client.h"
#include "third_party/raylib/include/raylib.h"

class CameraView {
public:
  CameraView(const std::string &name, VisionStreamType type);
  virtual ~CameraView();
  void draw(const Rectangle &rec);

protected:
  bool ensureConnection();

  std::unique_ptr<VisionIpcClient> client;
  Texture2D textureY = {};
  Texture2D textureUV = {};
  Shader shader = {};
  VisionBuf *frame = nullptr;
};
