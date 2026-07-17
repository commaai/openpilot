#include "tools/cabana/cameraview.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <utility>

#include <QApplication>

#include "common/yuv.h"
#include "imgui.h"
#include "imgui_impl_opengl3_loader.h"

CameraWidget::CameraWidget(std::string stream_name, VisionStreamType type, QWidget* parent) :
                          stream_name(stream_name), active_stream_type(type), requested_stream_type(type), ImGuiHost(parent) {
  setAttribute(Qt::WA_OpaquePaintEvent);
  qRegisterMetaType<std::set<VisionStreamType>>("availableStreams");
  QObject::connect(this, &CameraWidget::vipcThreadFrameReceived, this, &CameraWidget::vipcFrameReceived, Qt::QueuedConnection);
  QObject::connect(this, &CameraWidget::vipcAvailableStreamsUpdated, this, &CameraWidget::availableStreamsUpdated, Qt::QueuedConnection);
  QObject::connect(QApplication::instance(), &QCoreApplication::aboutToQuit, this, &CameraWidget::stopVipcThread);
}

CameraWidget::~CameraWidget() {
  stopVipcThread();
  if (texture && makeCurrent()) {
    glDeleteTextures(1, &texture);
  }
}

void CameraWidget::showEvent(QShowEvent *event) {
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

void CameraWidget::availableStreamsUpdated(std::set<VisionStreamType> streams) {
  available_streams = streams;
}

void CameraWidget::drawFrame() {
  {
    std::lock_guard lk(frame_lock);
    if (uploaded_gen != frame_gen) {
      uploaded_gen = frame_gen;
      tex_valid = !rgb_frame.empty();
      if (tex_valid) {
        if (!texture) glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        if (frame_width != tex_width || frame_height != tex_height) {
          glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
          glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
          glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
          glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
          glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
          glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, frame_width, frame_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgb_frame.data());
          tex_width = frame_width;
          tex_height = frame_height;
        } else {
          glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, tex_width, tex_height, GL_RGBA, GL_UNSIGNED_BYTE, rgb_frame.data());
        }
        glBindTexture(GL_TEXTURE_2D, 0);
      }
    }
  }
  if (!tex_valid) return;

  // aspect-fit letterbox
  const ImVec2 ds = ImGui::GetIO().DisplaySize;
  float widget_ratio = ds.x / ds.y;
  float frame_ratio = (float)tex_width / tex_height;
  float w = ds.x * std::min(frame_ratio / widget_ratio, 1.0f);
  float h = ds.y * std::min(widget_ratio / frame_ratio, 1.0f);
  ImVec2 p0((ds.x - w) / 2, (ds.y - h) / 2);
  ImVec2 uv0(0, 0), uv1(1, 1);
  if (active_stream_type == VISION_STREAM_DRIVER) std::swap(uv0.x, uv1.x);  // mirror driver camera
  ImGui::GetBackgroundDrawList()->AddImage(static_cast<ImTextureID>(texture), p0, ImVec2(p0.x + w, p0.y + h), uv0, uv1);
}

void CameraWidget::vipcFrameReceived() {
  update();
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
      emit vipcAvailableStreamsUpdated(streams);

      if (!vipc_client->connect(false)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        continue;
      }
    }

    if (VisionBuf *buf = vipc_client->recv(&frame_meta, 100)) {
      // NV12 -> RGBA once per frame on the receive thread; the GUI thread uploads to GL
      rgb_back.resize(buf->width * buf->height * 4);
      yuv::nv12_to_rgba(buf->y, buf->stride, buf->uv, buf->stride,
                        rgb_back.data(), buf->width * 4, buf->width, buf->height);
      {
        std::lock_guard lk(frame_lock);
        rgb_frame.swap(rgb_back);
        frame_width = buf->width;
        frame_height = buf->height;
        ++frame_gen;
      }
      emit vipcThreadFrameReceived();
    }
  }
}

// runs on the vipc thread: no GL here, just drop the CPU buffers and let the GUI thread notice
void CameraWidget::clearFrames() {
  std::lock_guard lk(frame_lock);
  rgb_frame.clear();
  rgb_back.clear();
  frame_width = frame_height = 0;
  ++frame_gen;
  available_streams.clear();
}
