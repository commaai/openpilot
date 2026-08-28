#include "tools/cabana/ui/widgets/cameraview.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>

#include "imgui_impl_opengl3_loader.h"

#include "common/yuv.h"
#include "tools/cabana/utils/util.h"

// GlTexture

void GlTexture::upload(const RgbImage &image) {
  if (id == 0) {
    glGenTextures(1, &id);
    glBindTexture(GL_TEXTURE_2D, id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
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

// CameraWidget

CameraWidget::CameraWidget(std::string stream_name, VisionStreamType type) :
                          stream_name(stream_name), active_stream_type(type), requested_stream_type(type) {
  // aboutToQuit -> stopVipcThread: the destructor runs before the GL/GLFW runtime is torn down
}

CameraWidget::~CameraWidget() {
  stopVipcThread();
}

void CameraWidget::showEvent() {
  if (!vipc_thread.joinable()) {
    clearFrames();
    vipc_exit = false;
    vipc_thread = std::thread(&CameraWidget::vipcThread, this);
  }
}

void CameraWidget::stopVipcThread() {
  vipc_exit = true;
  if (vipc_thread.joinable()) {
    vipc_thread.join();
  }
}

void CameraWidget::draw(const ImVec2 &size) {
  showEvent();
  ImGui::InvisibleButton("##camera", ImVec2(std::max(1.0f, size.x), std::max(1.0f, size.y)),
                         ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight | ImGuiButtonFlags_MouseButtonMiddle);
  rect_ = ImRect(ImGui::GetItemRectMin(), ImGui::GetItemRectMax());
  paintEvent();
  // mouseReleaseEvent
  if (ImGui::IsItemDeactivated()) clicked();
}

void CameraWidget::paintEvent() {
  ImDrawList *p = ImGui::GetWindowDrawList();
  p->AddRectFilled(rect_.Min, rect_.Max, bg);

  std::lock_guard lk(frame_lock);
  if (rgb_frame.isNull()) return;
  if (frame_updated) {
    frame_texture.upload(rgb_frame);
    frame_updated = false;
  }

  // Scale for aspect ratio
  float widget_ratio = (float)width() / height();
  float frame_ratio = (float)rgb_frame.width / rgb_frame.height;
  int w = std::lround(width() * std::min(frame_ratio / widget_ratio, 1.0f));
  int h = std::lround(height() * std::min(widget_ratio / frame_ratio, 1.0f));
  ImVec2 video_min(rect_.Min.x + (int)(width() - w) / 2, rect_.Min.y + (int)(height() - h) / 2);
  ImVec2 video_max(video_min.x + w, video_min.y + h);

  ImVec2 uv0(0, 0), uv1(1, 1);
  if (active_stream_type == VISION_STREAM_CABIN) {
    // mirror cabin camera horizontally
    uv0.x = 1;
    uv1.x = 0;
  }
  p->AddImage(frame_texture.ref(), video_min, video_max, uv0, uv1);
}

void CameraWidget::vipcThread() {
  VisionStreamType cur_stream = requested_stream_type;
  std::unique_ptr<VisionIpcClient> vipc_client;
  VisionIpcBufExtra frame_meta = {};

  while (!vipc_exit) {
    if (!vipc_client || cur_stream != requested_stream_type) {
      clearFrames();
      fprintf(stderr, "connecting to stream %d, was connected to %d\n",
              (int)requested_stream_type, (int)cur_stream);
      cur_stream = requested_stream_type;
      vipc_client.reset(new VisionIpcClient(stream_name, cur_stream, false));
    }
    active_stream_type = cur_stream;

    if (!vipc_client->connected) {
      clearFrames();
      auto streams = VisionIpcClient::getAvailableStreams(stream_name, false);
      if (streams.empty()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        continue;
      }
      utils::runOnMainThread([this, alive = std::weak_ptr<bool>(alive_), streams]() {
        if (alive.expired()) return;
        available_streams = streams;
        availableStreamsUpdated(streams);
      });

      if (!vipc_client->connect(false)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        continue;
      }
    }

    if (VisionBuf *buf = vipc_client->recv(&frame_meta, 100)) {
      // NV12 -> RGBA once per frame on the receive thread; paint just draws the image
      if (rgb_back.width != (int)buf->width || rgb_back.height != (int)buf->height) {
        rgb_back.resize(buf->width, buf->height);
      }
      yuv::nv12_to_rgba(buf->y, buf->stride, buf->uv, buf->stride,
                        rgb_back.data.data(), rgb_back.bytesPerLine(), buf->width, buf->height);
      {
        std::lock_guard lk(frame_lock);
        rgb_frame.swap(rgb_back);
        frame_updated = true;
      }
      // update(): imgui redraws every frame, the texture is uploaded in paintEvent
    }
  }
}

void CameraWidget::clearFrames() {
  std::lock_guard lk(frame_lock);
  rgb_frame.reset();
  rgb_back.reset();
  frame_updated = false;
  available_streams.clear();
}
