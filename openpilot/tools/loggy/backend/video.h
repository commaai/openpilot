#pragma once

#include "tools/loggy/backend/ingest.h"
#include "tools/loggy/backend/store.h"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace loggy {

enum class CameraViewKind : uint8_t {
  Road = 0,
  Driver = 1,
  WideRoad = 2,
  QRoad = 3,
};

struct CameraViewSpec {
  CameraViewKind view = CameraViewKind::Road;
  const char *label = "";
  const char *runtime_name = "";
  const char *layout_name = "";
  const char *encode_index = "";
};

struct CameraSegmentFile {
  int segment = -1;
  TimeRange range;
  std::string path;
};

struct CameraFrameIndexEntry {
  double timestamp = 0.0;
  int segment = -1;
  int decode_index = -1;
  uint32_t frame_id = 0;
  std::string path;
};

struct CameraDecodeKey {
  CameraViewKind view = CameraViewKind::Road;
  int segment = -1;
  int decode_index = -1;

  bool operator==(const CameraDecodeKey &other) const {
    return view == other.view && segment == other.segment && decode_index == other.decode_index;
  }
};

struct DecodedCameraFrame {
  CameraDecodeKey key;
  uint32_t frame_id = 0;
  double timestamp = 0.0;
  int width = 0;
  int height = 0;
  bool ok = false;
  std::string error;
  std::vector<uint8_t> rgba;
};

struct CameraDecodeStatus {
  bool has_source = false;
  bool loading = false;
  size_t cached_frames = 0;
  size_t queued_frames = 0;
  std::optional<CameraDecodeKey> active_key;
  std::optional<CameraDecodeKey> displayed_key;
  std::optional<CameraDecodeKey> failed_key;
  std::string error;
};

struct LiveCameraFrameStatus {
  bool supported = true;
  bool requested = false;
  bool connected = false;
  bool has_frame = false;
  size_t received_frames = 0;
  int width = 0;
  int height = 0;
  uint32_t frame_id = 0;
  std::string error;
};

struct CameraFeedIndex {
  CameraViewKind view = CameraViewKind::Road;
  std::vector<CameraSegmentFile> segment_files;
  std::vector<CameraFrameIndexEntry> entries;
};

const std::array<CameraViewSpec, 4> &camera_view_specs();
const CameraViewSpec &camera_view_spec(CameraViewKind view);
CameraViewKind camera_view_from_layout_name(std::string_view name,
                                            CameraViewKind fallback = CameraViewKind::Road);
std::string camera_view_layout_name(CameraViewKind view);
size_t camera_view_index(CameraViewKind view);
std::string camera_segment_path(const RouteSegment &segment, CameraViewKind view);
bool live_camera_view_supported(CameraViewKind view);
std::string live_camera_stream_label(CameraViewKind view);
CameraFeedIndex build_camera_feed_index(const std::vector<RouteSegment> &segments,
                                        const Store &store,
                                        CameraViewKind view,
                                        TimeRange range);
std::optional<CameraFrameIndexEntry> camera_frame_at_time(const CameraFeedIndex &index, double time);

class CameraFrameDecoder {
public:
  CameraFrameDecoder();
  ~CameraFrameDecoder();

  CameraFrameDecoder(const CameraFrameDecoder &) = delete;
  CameraFrameDecoder &operator=(const CameraFrameDecoder &) = delete;

  void set_camera_index(CameraFeedIndex index);
  // Same feed, more of it (ingest landed another segment): swap the index without discarding
  // the frame cache, warm FrameReader, or the displayed frame — set_camera_index resets all of
  // those, which restarted decode from scratch on every segment landing during route load.
  void update_camera_index(CameraFeedIndex index);
  void request_frame(double tracker_time);
  std::optional<DecodedCameraFrame> take_frame();
  // The consumer lost its uploaded texture (pane recreated by workspace undo/layout reload):
  // forget displayed_key_ so the next request_frame re-delivers instead of early-returning.
  void invalidate_displayed();
  CameraDecodeStatus status() const;

private:
  // pimpl: keeps FFmpeg/VisionIPC headers out of every pane include; allowed only here and panda_live (REVIEW §2.4)
  class Impl;
  std::unique_ptr<Impl> impl_;
};

class LiveCameraFrameSource {
public:
  explicit LiveCameraFrameSource(std::string server_name = "camerad");
  ~LiveCameraFrameSource();

  LiveCameraFrameSource(const LiveCameraFrameSource &) = delete;
  LiveCameraFrameSource &operator=(const LiveCameraFrameSource &) = delete;

  void set_enabled(bool enabled);
  void request_frame(CameraViewKind view);
  std::optional<DecodedCameraFrame> take_frame(CameraViewKind view);
  LiveCameraFrameStatus status(CameraViewKind view) const;

private:
  // pimpl: keeps FFmpeg/VisionIPC headers out of every pane include; allowed only here and panda_live (REVIEW §2.4)
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace loggy
