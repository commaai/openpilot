#pragma once
#include <memory>
#include <future>
#include <chrono>
#include "tools/cabana/streams/cerealseries.h"
#include "tools/cabana/ui/widgets/cameraview.h"
namespace cabana {
class RecordedCameraWidget {
public:
  RecordedCameraWidget();
  ~RecordedCameraWidget();
  void setSource(RecordedCameraReader reader);
  void draw(uint64_t time, const ImVec2 &size);
private:
  struct State;
  struct Result { RecordedFrame frame; RgbImage image; std::string error; };
  std::shared_ptr<State> state_;
  RecordedCameraReader reader_;
  std::future<Result> pending_;
  GlTexture texture_;
  std::optional<RecordedFrame> displayed_, requested_;
  std::string error_;
  std::chrono::steady_clock::time_point retry_at_;
};
}  // namespace cabana
