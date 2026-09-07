#include "tools/cabana/ui/widgets/cameraview.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <utility>

#include <GLFW/glfw3.h>
#include "imgui_impl_opengl3_loader.h"

#include "common/yuv.h"
#include "tools/cabana/utils/util.h"
#include "tools/cabana/settings.h"

namespace {
constexpr GLenum GL_LINEAR_MIPMAP_LINEAR_ = 0x2703;
// glGenerateMipmap is not part of the imgui GL loader
void generateMipmap() {
  static auto fn = (void (*)(GLenum))glfwGetProcAddress("glGenerateMipmap");
  if (fn) fn(GL_TEXTURE_2D);
}
}  // namespace

void GlTexture::upload(const RgbImage &image) {
  if (id == 0) {
    glGenTextures(1, &id);
    glBindTexture(GL_TEXTURE_2D, id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, mipmap ? GL_LINEAR_MIPMAP_LINEAR_ : GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  } else {
    glBindTexture(GL_TEXTURE_2D, id);
  }
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
  if (width != image.width || height != image.height) {
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.width, image.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image.data.data());
    width = image.width;
    height = image.height;
  } else {
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, image.data.data());
  }
  if (mipmap) generateMipmap();
  glBindTexture(GL_TEXTURE_2D, 0);
}

void GlTexture::destroy() {
  if (id != 0) {
    glDeleteTextures(1, &id);
  }
  id = 0;
  width = height = 0;
  key = 0;
}

CameraWidget::CameraWidget(std::string stream_name, VisionStreamType type)
    : stream_name_(stream_name), active_stream_type_(type), requested_stream_type_(type) {}

CameraWidget::~CameraWidget() {
  stopVipcThread();
}

void CameraWidget::startVipcThread() {
  if (!vipc_thread_.joinable()) {
    // Preserve the last frame when restoring a collapsed video; paused replay sends no replacement.
    vipc_exit_ = false;
    vipc_thread_ = std::thread(&CameraWidget::vipcThread, this);
  }
}

void CameraWidget::stopVipcThread() {
  vipc_exit_ = true;
  if (vipc_thread_.joinable()) {
    vipc_thread_.join();
  }
}

void CameraWidget::setVisible(bool visible) {
  if (visible == visible_) return;
  visible_ = visible;
  visible ? startVipcThread() : stopVipcThread();
}

void CameraWidget::draw(const ImVec2 &size) {
  setVisible(true);
  ImGui::InvisibleButton("##camera", ImVec2(std::max(1.0f, size.x), std::max(1.0f, size.y)),
                         ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight | ImGuiButtonFlags_MouseButtonMiddle);
  rect_ = ImRect(ImGui::GetItemRectMin(), ImGui::GetItemRectMax());
  paint();
  if (ImGui::IsItemDeactivated()) clicked();
}

float CameraWidget::frameAspectRatio() const {
  if (frame_texture_.width > 0 && frame_texture_.height > 0) {
    return (float)frame_texture_.width / frame_texture_.height;
  }
  return 1928.0f / 1208.0f;  // the road camera, until the first frame arrives
}

void CameraWidget::paint() {
  ImDrawList *p = ImGui::GetWindowDrawList();
  p->AddRectFilled(rect_.Min, rect_.Max, bg_, ImGui::GetStyle().ChildRounding);

  std::lock_guard lk(frame_lock_);
  if (rgb_frame_.isNull()) return;
  if (frame_updated_) {
    frame_texture_.upload(rgb_frame_);
    frame_updated_ = false;
  }

  VideoPlacement placement = videoPlacement(rect_, frameAspectRatio(), settings.crop_video);
  if (active_stream_type_ == VISION_STREAM_CABIN) {
    // mirror cabin camera horizontally
    std::swap(placement.uv0.x, placement.uv1.x);
  }
  p->AddImageRounded(frame_texture_.ref(), placement.min, placement.max, placement.uv0, placement.uv1, IM_COL32_WHITE, ImGui::GetStyle().ChildRounding);
}

void CameraWidget::vipcThread() {
  VisionStreamType cur_stream = requested_stream_type_;
  std::unique_ptr<VisionIpcClient> vipc_client;
  VisionIpcBufExtra frame_meta = {};
  bool was_connected = false;

  while (!vipc_exit_) {
    if (!vipc_client || cur_stream != requested_stream_type_) {
      if (cur_stream != requested_stream_type_) clearFrames();
      cur_stream = requested_stream_type_;
      vipc_client.reset(new VisionIpcClient(stream_name_, cur_stream, false));
    }
    active_stream_type_ = cur_stream;

    if (!vipc_client->connected) {
      // the server changed (a new route): the last frame is stale. A fresh thread keeps it, see startVipcThread().
      if (std::exchange(was_connected, false)) clearFrames();
      auto streams = VisionIpcClient::getAvailableStreams(stream_name_, false);
      if (streams.empty()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        continue;
      }
      utils::runOnMainThread(utils::guarded(alive_, [this, streams]() { availableStreamsUpdated(streams); }));

      if (!vipc_client->connect(false)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        continue;
      }
      was_connected = true;
    }

    if (VisionBuf *buf = vipc_client->recv(&frame_meta, 100)) {
      // NV12 -> RGBA once per frame on the receive thread; paint just draws the image
      if (rgb_back_.width != (int)buf->width || rgb_back_.height != (int)buf->height) {
        rgb_back_.resize(buf->width, buf->height);
      }
      yuv::nv12_to_rgba(buf->y, buf->stride, buf->uv, buf->stride,
                        rgb_back_.data.data(), rgb_back_.bytesPerLine(), buf->width, buf->height);
      {
        std::lock_guard lk(frame_lock_);
        rgb_frame_.swap(rgb_back_);
        frame_updated_ = true;
      }
    }
  }
}

void CameraWidget::clearFrames() {
  std::lock_guard lk(frame_lock_);
  rgb_frame_.reset();
  rgb_back_.reset();
  frame_updated_ = false;
}
