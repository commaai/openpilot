// ImGui port of tools/cabana/cameraview.cc (CameraWidget) + the camera-view
// slice of tools/cabana/videowidget.cc (StreamCameraView, camera tabs).
// Timeline/thumbnails/alert overlays are P5.2's job -- see MIGRATION.md
// Phase 5. Ported 1:1 from those frozen Qt sources unless noted below.
#include "tools/cabana/imgui/app.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

#include "imgui_impl_opengl3_loader.h"
#include "libyuv.h"

#include "msgq/visionipc/visionipc_client.h"

namespace {

// Indexed by VisionStreamType (ROAD=0, DRIVER=1, WIDE_ROAD=2) -- mirrors
// VideoWidget::vipcAvailableStreamsUpdated()'s stream_names array. VISION_STREAM_MAP
// is deliberately excluded (Qt's array would be out-of-bounds for it too; we
// just skip it defensively instead of replicating the UB).
constexpr const char *kStreamNames[3] = {"Road camera", "Driver camera", "Wide road camera"};

// Hand-rolled tab button instead of ImGui::BeginTabBar()/BeginTabItem(): see
// detail_panel.cc's draw_tab_button() comment -- native tab-item caption text
// doesn't render on the frame a tab is (re)created in this ImGui build, which
// would show blank camera tab labels in the mandatory single-shot --output
// capture. Same Button + Tab/TabSelected/TabHovered theme colors as that helper.
bool draw_camera_tab_button(const char *label, bool selected) {
  ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(selected ? ImGuiCol_TabSelected : ImGuiCol_Tab));
  ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImGui::GetStyleColorVec4(ImGuiCol_TabHovered));
  ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImGui::GetStyleColorVec4(ImGuiCol_TabSelected));
  ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(selected ? ImGuiCol_Text : ImGuiCol_TextDisabled));
  const bool clicked = ImGui::Button(label);
  ImGui::PopStyleColor(4);
  return clicked;
}

// Owns the background VisionIpcClient thread + the GL texture it feeds.
// Frame handoff mirrors CameraWidget's design (cameraview.cc): the vipc
// thread stores a raw VisionBuf* + meta under a mutex; the UI thread converts
// NV12 -> RGBA (libyuv, jotpluggler's proven CameraFeedView pattern) and
// uploads to GL once per new frame id, entirely off the vipc thread.
//
// Lifetime note: this is a function-local static in draw_video_panel(), so
// (unlike AppState, which is destroyed inside run() while the GL context is
// still current) its destructor runs at process atexit time, *after*
// GlfwRuntime has already torn down the GLFW/GL context -- app.cc/app.h/main.cc
// are off-limits for this workstream so there's no hook to tear this down
// earlier. See the destructor below for how that's handled.
class CameraViewState {
 public:
  CameraViewState() = default;
  // Deliberately does *not* call glDeleteTextures() here: this is a
  // function-local static, so its destructor runs at process atexit time --
  // strictly after GlfwRuntime::~GlfwRuntime() has already called
  // glfwDestroyWindow()/glfwTerminate() (run()'s locals, incl. GlfwRuntime,
  // unwind on normal return from run(), which happens before atexit-driven
  // static destructors ever start; app.cc/app.h are off-limits for this
  // workstream so there's no earlier hook to delete the texture while the
  // context is still current, the way jotpluggler's run() explicitly resets
  // its CameraFeedViews before falling out of scope). Calling any GL/GLFW
  // function against a torn-down context here would be the actual bug; one
  // leaked texture handle at process exit is reclaimed by the OS/driver like
  // every other GPU resource the process held, so it's not a real leak.
  // Only the vipc thread (which owns no GL state) needs an explicit,
  // guaranteed-safe shutdown -- see stop().
  ~CameraViewState() { stop(); }
  CameraViewState(const CameraViewState &) = delete;
  CameraViewState &operator=(const CameraViewState &) = delete;

  void ensure_running() {
    if (thread_.joinable()) return;
    stop_requested_.store(false);
    thread_ = std::thread([this] { vipc_thread(); });
  }

  // Bounded by the vipc thread's 100ms recv/retry timeout (matches
  // CameraWidget::stopVipcThread()'s QThread::wait() after
  // requestInterruption() -- no hang, ~100ms worst case).
  void stop() {
    if (!thread_.joinable()) return;
    stop_requested_.store(true);
    thread_.join();
    clear_frame();
  }

  // NV12 -> RGBA convert + GL upload for the latest frame, if any and if it's
  // new. Only call while the Video panel is actually visible (perf: no
  // conversion work for a hidden/collapsed dock tab).
  void sync_texture() {
    uint32_t frame_id = 0;
    int w = 0, h = 0, stride = 0;
    bool need_upload = false;
    {
      std::lock_guard<std::mutex> lk(frame_mutex_);
      if (current_frame_ == nullptr) return;
      frame_id = frame_meta_.frame_id;
      if (frame_id == last_uploaded_frame_id_) return;  // unchanged: skip conversion entirely
      w = stream_width_;
      h = stream_height_;
      stride = stream_stride_;
      if (w <= 0 || h <= 0 || stride <= 0) return;
      rgba_scratch_.resize(static_cast<size_t>(w) * static_cast<size_t>(h) * 4U);
      // Proven NV12->ABGR upload pattern from tools/jotpluggler/runtime.cc
      // CameraFeedView (ABGR-named output == RGBA byte order in memory,
      // matching the GL_RGBA texture format below).
      libyuv::NV12ToABGR(current_frame_->y, stride, current_frame_->uv, stride,
                          rgba_scratch_.data(), w * 4, w, h);
      need_upload = true;
    }
    if (!need_upload) return;
    last_uploaded_frame_id_ = frame_id;
    upload_gl(w, h);
  }

  // QTabBar::autoHide equivalent: hidden entirely below 2 streams.
  void draw_tabs() {
    std::set<VisionStreamType> streams;
    {
      std::lock_guard<std::mutex> lk(frame_mutex_);
      streams = available_streams_;
    }
    if (streams.size() < 2) return;

    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(2.0f, ImGui::GetStyle().ItemSpacing.y));
    bool first = true;
    for (VisionStreamType type : streams) {
      if (!first) ImGui::SameLine();
      first = false;
      const char *label = (type >= VISION_STREAM_ROAD && type <= VISION_STREAM_WIDE_ROAD) ? kStreamNames[type] : "Camera";
      ImGui::PushID(static_cast<int>(type));
      if (draw_camera_tab_button(label, type == requested_type_.load())) {
        requested_type_.store(type);
      }
      ImGui::PopID();
    }
    ImGui::PopStyleVar();
    ImGui::Spacing();
  }

  // Aspect-fit centered in `avail`, black letterboxing -- mirrors
  // CameraWidget::paintGL()'s scale_x/scale_y quad transform, expressed as a
  // fit rect instead of a shader uniform since we composite via the ImGui
  // draw list rather than a raw GL viewport. Clicking the frame toggles
  // pause, mirroring CameraWidget::clicked() -> VideoWidget's
  // can->pause(!isPaused()) wiring.
  void draw_frame(AppState &app, ImVec2 avail) {
    avail.x = std::max(1.0f, avail.x);
    avail.y = std::max(1.0f, avail.y);
    const ImVec2 top_left = ImGui::GetCursorScreenPos();

    ImGui::InvisibleButton("##camera_frame", avail);
    if (ImGui::IsItemClicked()) {
      app.stream->pause(!app.stream->isPaused());
    }

    ImDrawList *draw_list = ImGui::GetWindowDrawList();
    draw_list->AddRectFilled(top_left, ImVec2(top_left.x + avail.x, top_left.y + avail.y), IM_COL32(0, 0, 0, 255));

    if (texture_ != 0 && texture_width_ > 0 && texture_height_ > 0) {
      const float frame_ratio = static_cast<float>(texture_width_) / static_cast<float>(texture_height_);
      const float widget_ratio = avail.x / avail.y;
      ImVec2 size = avail;
      if (widget_ratio > frame_ratio) {
        size.x = avail.y * frame_ratio;
      } else {
        size.y = avail.x / frame_ratio;
      }
      const ImVec2 origin(top_left.x + (avail.x - size.x) * 0.5f, top_left.y + (avail.y - size.y) * 0.5f);
      draw_list->AddImage(static_cast<ImTextureID>(texture_), origin, ImVec2(origin.x + size.x, origin.y + size.y));
    }
  }

 private:
  void clear_frame() {
    std::lock_guard<std::mutex> lk(frame_mutex_);
    current_frame_ = nullptr;
    available_streams_.clear();
  }

  void upload_gl(int w, int h) {
    const bool new_size = (texture_ == 0) || texture_width_ != w || texture_height_ != h;
    if (texture_ == 0) glGenTextures(1, &texture_);
    glBindTexture(GL_TEXTURE_2D, texture_);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    if (new_size) {
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba_scratch_.data());
      texture_width_ = w;
      texture_height_ = h;
    } else {
      glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, rgba_scratch_.data());
    }
    glBindTexture(GL_TEXTURE_2D, 0);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);  // restore default so ImGui's own texture uploads aren't affected
  }

  // Mirrors CameraWidget::vipcThread(): reconnect loop + conflate=false, incl.
  // reconnecting when the requested stream type changes (camera tab click).
  void vipc_thread() {
    VisionStreamType cur_stream = requested_type_.load();
    std::unique_ptr<VisionIpcClient> vipc_client;

    while (!stop_requested_.load()) {
      if (!vipc_client || cur_stream != requested_type_.load()) {
        clear_frame();
        cur_stream = requested_type_.load();
        vipc_client = std::make_unique<VisionIpcClient>("camerad", cur_stream, /*conflate=*/false);
      }

      if (!vipc_client->connected) {
        clear_frame();
        std::set<VisionStreamType> streams = VisionIpcClient::getAvailableStreams("camerad", /*blocking=*/false);
        if (streams.empty()) {
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
          continue;
        }
        {
          std::lock_guard<std::mutex> lk(frame_mutex_);
          available_streams_ = streams;
        }

        if (!vipc_client->connect(/*blocking=*/false)) {
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
          continue;
        }
        std::lock_guard<std::mutex> lk(frame_mutex_);
        stream_width_ = static_cast<int>(vipc_client->buffers[0].width);
        stream_height_ = static_cast<int>(vipc_client->buffers[0].height);
        stream_stride_ = static_cast<int>(vipc_client->buffers[0].stride);
      }

      VisionIpcBufExtra meta{};
      if (VisionBuf *buf = vipc_client->recv(&meta, 100)) {
        std::lock_guard<std::mutex> lk(frame_mutex_);
        current_frame_ = buf;
        frame_meta_ = meta;
      }
    }

    // Drop the pointer into vipc_client's buffers before it (and they) are
    // destroyed below, so a UI-thread sync_texture() racing the very end of
    // shutdown never dereferences a freed buffer.
    clear_frame();
  }

  std::thread thread_;
  std::atomic<bool> stop_requested_{false};
  std::atomic<VisionStreamType> requested_type_{VISION_STREAM_ROAD};

  // Shared with vipc_thread(); guarded by frame_mutex_.
  std::mutex frame_mutex_;
  VisionBuf *current_frame_ = nullptr;
  VisionIpcBufExtra frame_meta_{};
  int stream_width_ = 0;
  int stream_height_ = 0;
  int stream_stride_ = 0;
  std::set<VisionStreamType> available_streams_;

  // UI-thread only.
  uint32_t last_uploaded_frame_id_ = std::numeric_limits<uint32_t>::max();
  std::vector<uint8_t> rgba_scratch_;
  GLuint texture_ = 0;
  int texture_width_ = 0;
  int texture_height_ = 0;
};

}  // namespace

void draw_video_panel(AppState &app) {
  static CameraViewState state;

  // Mirrors VideoWidget only building a camera widget when
  // !can->liveStreaming() -- i.e. a route replay. DummyStream (no stream) and
  // future live sources (panda/socketcan/msgq/zmq -- not wired up yet, see
  // main.cc) default liveStreaming()=true and show no camera pane in Qt either.
  const bool want_camera = has_stream(app) && !app.stream->liveStreaming();
  if (want_camera) {
    state.ensure_running();
  } else {
    state.stop();
  }

  if (ImGui::Begin(VIDEO_WINDOW_TITLE)) {
    // No stream / DummyStream / no frames yet all fall out of the same empty
    // state here: available_streams_ empty (no tabs) and texture_ == 0 (plain
    // black fill in draw_frame), matching CameraWidget::paintGL's disconnected
    // look without a separate code path.
    state.sync_texture();
    state.draw_tabs();
    state.draw_frame(app, ImGui::GetContentRegionAvail());
  }
  ImGui::End();
}
