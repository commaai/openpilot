#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "imgui.h"
#include "imgui_internal.h"
#include "openpilot/cereal/visionstream.h"
#include "tools/cabana/core/observable.h"
#include "msgq/visionipc/visionipc_client.h"

// tightly packed RGBA pixels
struct RgbImage {
  int width = 0;
  int height = 0;
  std::vector<uint8_t> data;
  bool isNull() const { return data.empty(); }
  void reset() { width = height = 0; data.clear(); }
  void resize(int w, int h) { width = w; height = h; data.resize(size_t(w) * h * 4); }
  int bytesPerLine() const { return width * 4; }
  void swap(RgbImage &other) { std::swap(width, other.width); std::swap(height, other.height); data.swap(other.data); }
};

// GL texture holding an RgbImage. Created/uploaded/freed on the GUI thread only (the GL context is current there).
struct GlTexture {
  GlTexture() = default;
  GlTexture(const GlTexture &) = delete;
  GlTexture &operator=(const GlTexture &) = delete;
  ~GlTexture() { destroy(); }
  void upload(const RgbImage &image);  // (re)allocates when the size changes
  void destroy();
  ImTextureRef ref() const { return ImTextureRef((ImTextureID)(uintptr_t)id); }

  unsigned int id = 0;
  int width = 0;
  int height = 0;
  uint64_t key = 0;  // caller-defined identity of the uploaded image
  bool mipmap = false;  // set before the first upload: a mip chain for images drawn downscaled
};

class CameraWidget {
public:
  explicit CameraWidget(std::string stream_name, VisionStreamType stream_type);
  ~CameraWidget();
  void setStreamType(VisionStreamType type) { requested_stream_type_ = type; }
  void stopVipcThread();
  // draw() implies visible; the owner calls setVisible(false) when the widget is no longer drawn, which
  // stops the vipc thread.
  void setVisible(bool visible);
  // draws an item of `size` into the current window
  void draw(const ImVec2 &size);
  const ImRect &rect() const { return rect_; }
  float frameAspectRatio() const;
  float width() const { return rect_.GetWidth(); }
  float height() const { return rect_.GetHeight(); }

  Observable<> clicked;
  Observable<std::set<VisionStreamType>> availableStreamsUpdated;  // invoked on the main thread

private:
  void paint();
  void startVipcThread();
  void vipcThread();
  void clearFrames();

  ImU32 bg_ = IM_COL32(0, 0, 0, 255);
  RgbImage rgb_frame_;   // written by vipc thread, drawn by GUI thread; guarded by frame_lock_
  RgbImage rgb_back_;    // vipc thread only
  bool frame_updated_ = false;  // rgb_frame_ changed since the last upload; guarded by frame_lock_
  GlTexture frame_texture_;     // GUI thread only
  ImRect rect_;
  bool visible_ = false;

  std::string stream_name_;
  std::atomic<VisionStreamType> active_stream_type_;
  std::atomic<VisionStreamType> requested_stream_type_;
  std::thread vipc_thread_;
  std::atomic<bool> vipc_exit_ = false;
  std::mutex frame_lock_;
  std::shared_ptr<bool> alive_ = std::make_shared<bool>(true);
};
