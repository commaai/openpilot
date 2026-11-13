#include "tools/cabana/ui/chart/recordedcamera.h"
#include "tools/cabana/ui/threadpool.h"
#include "tools/replay/framereader.h"
#include "common/yuv.h"
#include "system/camerad/cameras/nv12_info.h"
#include <atomic>
#include <chrono>
#include <stdexcept>

namespace cabana {
struct RecordedCameraWidget::State {
  std::atomic<bool> abort{false};
  std::unique_ptr<FrameReader> reader;
  std::string path;
};
RecordedCameraWidget::RecordedCameraWidget() : state_(std::make_shared<State>()) {}
RecordedCameraWidget::~RecordedCameraWidget() { state_->abort = true; }
void RecordedCameraWidget::setSource(RecordedCameraReader reader) { reader_ = std::move(reader); }
void RecordedCameraWidget::draw(uint64_t time, const ImVec2 &size) {
  const auto desired = reader_ ? reader_(time) : std::nullopt;
  if (pending_.valid() && pending_.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
    auto result = pending_.get();
    error_ = result.error;
    retry_at_ = error_.empty() ? std::chrono::steady_clock::time_point{} : std::chrono::steady_clock::now() + std::chrono::seconds(2);
    if (desired && result.frame.path == desired->path && result.frame.mono_time <= time &&
        time - result.frame.mono_time < (result.frame.path == "thumbnail" ? 30000000000ULL : 1000000000ULL) && !result.image.isNull()) {
      texture_.upload(result.image);
      displayed_ = result.frame;
    }
    requested_.reset();
  }
  if (desired && std::chrono::steady_clock::now() >= retry_at_ && !pending_.valid() && (!displayed_ || displayed_->path != desired->path || displayed_->index != desired->index || displayed_->mono_time != desired->mono_time)) {
    const auto frame = *desired;
    requested_ = frame;
    pending_ = ThreadPool::media().run([state = state_, frame]() {
      Result result; result.frame = frame;
      try {
        if (state->abort) return result;
        if (frame.jpeg && !frame.jpeg->empty()) {
          if (!decodeJpegImage(frame.jpeg->data(), frame.jpeg->size(), &result.image)) throw std::runtime_error("Cannot decode thumbnail");
          return result;
        }
        if (state->path != frame.path) {
          auto reader = std::make_unique<FrameReader>();
          if (!reader->load(static_cast<CameraType>(frame.camera), frame.path, true, &state->abort, true)) {
            throw std::runtime_error("Camera segment unavailable");
          }
          state->reader = std::move(reader); state->path = frame.path;
        }
        if (state->abort) return result;
        const auto [stride, y_height, uv_height, bytes] = get_nv12_info(state->reader->width, state->reader->height);
        (void)uv_height;
        VisionBuf buffer;
        buffer.allocate(bytes);
        buffer.init_yuv(state->reader->width, state->reader->height, stride, stride * y_height);
        struct Release { VisionBuf *buffer; ~Release() { buffer->free(); } } release{&buffer};
        if (!state->reader->get(frame.index, &buffer)) throw std::runtime_error("Cannot decode camera frame");
        result.image.resize(state->reader->width, state->reader->height);
        yuv::nv12_to_rgba(buffer.y, buffer.stride, buffer.uv, buffer.stride, result.image.data.data(),
                         result.image.bytesPerLine(), result.image.width, result.image.height);
      } catch (const std::exception &error) { result.error = error.what(); }
      return result;
    });
  }
  const auto position = ImGui::GetCursorScreenPos();
  ImGui::InvisibleButton("recorded_camera", ImVec2(std::max(size.x, 1.0f), std::max(size.y, 1.0f)));
  const ImRect rect(position, ImGui::GetItemRectMax());
  if (displayed_ && displayed_->mono_time <= time && time - displayed_->mono_time < (displayed_->path == "thumbnail" ? 30000000000ULL : 1000000000ULL) && texture_.id) {
    const auto placement = videoPlacement(rect, static_cast<float>(texture_.width) / texture_.height, false);
    drawVideoFrame(ImGui::GetWindowDrawList(), texture_.ref(), rect, placement);
  } else ImGui::GetWindowDrawList()->AddText(position, ImGui::GetColorU32(ImGuiCol_TextDisabled),
              error_.empty() ? (desired ? "Loading camera frame..." : "No camera frame at playback time") : error_.c_str());
}
}  // namespace cabana
