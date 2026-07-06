#include "tools/loggy/backend/video.h"

#include "libyuv.h"
#include "msgq/visionipc/visionipc_client.h"
#include "system/camerad/cameras/nv12_info.h"
#include "tools/replay/framereader.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <unordered_map>
#include <utility>

namespace loggy {
namespace {

int rounded_int(double value) {
  return static_cast<int>(std::llround(value));
}

uint32_t rounded_frame_id(double value) {
  return static_cast<uint32_t>(std::max(0, rounded_int(value)));
}

const SeriesView encode_series_or_empty(const Store &store, const std::string &path, TimeRange range) {
  return path.empty() ? SeriesView{} : store.series_full(path, range);
}

CameraType decoder_camera_type(CameraViewKind view) {
  switch (view) {
    case CameraViewKind::Driver: return DriverCam;
    case CameraViewKind::WideRoad: return WideRoadCam;
    case CameraViewKind::QRoad: return RoadCam;
    case CameraViewKind::Road:
    default: return RoadCam;
  }
}

std::optional<VisionStreamType> live_camera_stream_type(CameraViewKind view) {
  switch (view) {
    case CameraViewKind::Road: return VISION_STREAM_ROAD;
    case CameraViewKind::Driver: return VISION_STREAM_DRIVER;
    case CameraViewKind::WideRoad: return VISION_STREAM_WIDE_ROAD;
    case CameraViewKind::QRoad:
    default: return std::nullopt;
  }
}

struct DecodeRequest {
  CameraDecodeKey key;
  uint32_t frame_id = 0;
  double timestamp = 0.0;
  std::string path;
  uint64_t serial = 0;
  uint64_t generation = 0;
  bool display = true;
};

bool same_key(const std::optional<CameraDecodeKey> &lhs, const CameraDecodeKey &rhs) {
  return lhs.has_value() && *lhs == rhs;
}

bool same_key(const CameraDecodeKey &lhs, const CameraDecodeKey &rhs) {
  return lhs == rhs;
}

size_t nearest_frame_index(const CameraFeedIndex &index, double time) {
  if (index.entries.empty()) return 0;
  const auto upper = std::lower_bound(index.entries.begin(), index.entries.end(), time,
                                      [](const CameraFrameIndexEntry &entry, double value) {
    return entry.timestamp < value;
  });
  if (upper == index.entries.begin()) return 0;
  if (upper == index.entries.end()) return index.entries.size() - 1;
  const size_t next_index = static_cast<size_t>(upper - index.entries.begin());
  const CameraFrameIndexEntry &next = *upper;
  const CameraFrameIndexEntry &prev = *(upper - 1);
  return std::abs(next.timestamp - time) < std::abs(time - prev.timestamp) ? next_index : next_index - 1;
}

uint32_t clamp_frame_id(uint64_t frame_id) {
  return static_cast<uint32_t>(std::min<uint64_t>(frame_id, std::numeric_limits<uint32_t>::max()));
}

int clamp_decode_index(uint64_t frame_id) {
  return static_cast<int>(std::min<uint64_t>(frame_id, static_cast<uint64_t>(std::numeric_limits<int>::max())));
}

std::string segment_path_for(const CameraFeedIndex &index, int segment) {
  for (const CameraSegmentFile &file : index.segment_files) {
    if (file.segment == segment) return file.path;
  }
  return {};
}

DecodeRequest request_for_entry(const CameraFeedIndex &index,
                                const CameraFrameIndexEntry &frame,
                                uint64_t serial,
                                uint64_t generation,
                                bool display) {
  return DecodeRequest{
    .key = CameraDecodeKey{.view = index.view, .segment = frame.segment, .decode_index = frame.decode_index},
    .frame_id = frame.frame_id,
    .timestamp = frame.timestamp,
    .path = segment_path_for(index, frame.segment),
    .serial = serial,
    .generation = generation,
    .display = display,
  };
}

}  // namespace

const std::array<CameraViewSpec, 4> &camera_view_specs() {
  static const std::array<CameraViewSpec, 4> specs = {{
    {CameraViewKind::Road, "Road Camera", "road", "road", "roadEncodeIdx"},
    {CameraViewKind::Driver, "Driver Camera", "driver", "driver", "driverEncodeIdx"},
    {CameraViewKind::WideRoad, "Wide Road Camera", "wide", "wide_road", "wideRoadEncodeIdx"},
    {CameraViewKind::QRoad, "qRoad Camera", "qroad", "qroad", "qRoadEncodeIdx"},
  }};
  return specs;
}

size_t camera_view_index(CameraViewKind view) {
  const int raw = static_cast<int>(view);
  return raw >= 0 && raw < static_cast<int>(camera_view_specs().size()) ? static_cast<size_t>(raw) : 0U;
}

const CameraViewSpec &camera_view_spec(CameraViewKind view) {
  return camera_view_specs()[camera_view_index(view)];
}

CameraViewKind camera_view_from_layout_name(std::string_view name, CameraViewKind fallback) {
  for (const CameraViewSpec &spec : camera_view_specs()) {
    if (name == spec.layout_name || name == spec.runtime_name) return spec.view;
  }
  return fallback;
}

std::string camera_view_layout_name(CameraViewKind view) {
  return camera_view_spec(view).layout_name;
}

std::string camera_segment_path(const RouteSegment &segment, CameraViewKind view) {
  switch (view) {
    case CameraViewKind::Driver: return segment.driver_camera_path;
    case CameraViewKind::WideRoad: return segment.wide_road_camera_path;
    case CameraViewKind::QRoad: return segment.qroad_camera_path;
    case CameraViewKind::Road:
    default: return segment.road_camera_path;
  }
}

bool live_camera_view_supported(CameraViewKind view) {
  return live_camera_stream_type(view).has_value();
}

std::string live_camera_stream_label(CameraViewKind view) {
  return camera_view_spec(view).runtime_name;
}

CameraFeedIndex build_camera_feed_index(const std::vector<RouteSegment> &segments,
                                        const Store &store,
                                        CameraViewKind view,
                                        TimeRange range) {
  CameraFeedIndex index;
  index.view = view;

  std::unordered_map<int, std::string> segment_paths;
  segment_paths.reserve(segments.size());
  for (const RouteSegment &segment : segments) {
    const std::string path = camera_segment_path(segment, view);
    if (path.empty()) continue;
    index.segment_files.push_back(CameraSegmentFile{
      .segment = segment.segment,
      .range = segment.range,
      .path = path,
    });
    segment_paths[segment.segment] = path;
  }
  if (segment_paths.empty()) return index;

  const std::string prefix = "/" + std::string(camera_view_spec(view).encode_index);
  const SeriesView segment_numbers = encode_series_or_empty(store, prefix + "/segmentNum", range);
  SeriesView decode_indices = encode_series_or_empty(store, prefix + "/segmentId", range);
  if (decode_indices.points.empty()) {
    decode_indices = encode_series_or_empty(store, prefix + "/segmentIdEncode", range);
  }
  const SeriesView frame_ids = encode_series_or_empty(store, prefix + "/frameId", range);
  if (segment_numbers.points.empty() || decode_indices.points.empty()) return index;

  size_t count = std::min(segment_numbers.points.size(), decode_indices.points.size());
  if (!frame_ids.points.empty()) count = std::min(count, frame_ids.points.size());
  index.entries.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    const int segment_number = rounded_int(segment_numbers.points[i].value);
    if (segment_paths.find(segment_number) == segment_paths.end()) continue;
    const int decode_index = rounded_int(decode_indices.points[i].value);
    const uint32_t frame_id = !frame_ids.points.empty()
      ? rounded_frame_id(frame_ids.points[i].value)
      : static_cast<uint32_t>(std::max(0, decode_index));
    index.entries.push_back(CameraFrameIndexEntry{
      .timestamp = segment_numbers.points[i].t,
      .segment = segment_number,
      .decode_index = decode_index,
      .frame_id = frame_id,
    });
  }

  std::sort(index.entries.begin(), index.entries.end(), [](const CameraFrameIndexEntry &a,
                                                           const CameraFrameIndexEntry &b) {
    if (std::abs(a.timestamp - b.timestamp) > 1.0e-9) return a.timestamp < b.timestamp;
    if (a.segment != b.segment) return a.segment < b.segment;
    return a.decode_index < b.decode_index;
  });
  return index;
}

std::optional<CameraFrameIndexEntry> camera_frame_at_time(const CameraFeedIndex &index, double time) {
  if (index.entries.empty()) return std::nullopt;
  const auto upper = std::lower_bound(index.entries.begin(), index.entries.end(), time,
                                      [](const CameraFrameIndexEntry &entry, double value) {
    return entry.timestamp < value;
  });
  if (upper == index.entries.begin()) return index.entries.front();
  if (upper == index.entries.end()) return index.entries.back();
  const CameraFrameIndexEntry &next = *upper;
  const CameraFrameIndexEntry &prev = *(upper - 1);
  return std::abs(next.timestamp - time) < std::abs(time - prev.timestamp) ? next : prev;
}

class CameraFrameDecoder::Impl {
public:
  Impl() : worker_([this]() { workerLoop(); }) {}

  ~Impl() {
    stop_.store(true);
    abort_.store(true);
    cv_.notify_all();
    if (worker_.joinable()) worker_.join();
  }

  void set_camera_index(CameraFeedIndex index) {
    uint64_t generation = 0;
    {
      std::lock_guard lock(mutex_);
      index_ = std::move(index);
      generation = ++generation_;
      ++latest_serial_;
      queued_requests_.clear();
      pending_result_.reset();
      cache_.clear();
      active_key_.reset();
      displayed_key_.reset();
      failed_key_.reset();
      error_.clear();
      loading_ = false;
    }
    // Abort BEFORE touching reader_mutex_: an in-flight FrameReader::load holds that lock for
    // the whole segment fetch (seconds for a remote file), and load polls abort_ — raising it
    // first is what makes the acquire below return promptly instead of stalling the UI thread.
    abort_.store(true);
    cv_.notify_all();
    {
      std::lock_guard reader_lock(reader_mutex_);
      reader_.reset();
      reader_segment_ = -1;
      reader_generation_ = generation;
    }
  }

  void update_camera_index(CameraFeedIndex index) {
    std::lock_guard lock(mutex_);
    index_ = std::move(index);
  }

  void request_frame(double tracker_time) {
    bool queued = false;
    {
      std::lock_guard lock(mutex_);
      if (index_.entries.empty()) return;

      const size_t frame_index = nearest_frame_index(index_, tracker_time);
      const CameraFrameIndexEntry &frame = index_.entries[frame_index];
      const CameraDecodeKey key{.view = index_.view, .segment = frame.segment, .decode_index = frame.decode_index};
      if (same_key(displayed_key_, key) ||
          same_key(active_key_, key) ||
          same_key(failed_key_, key)) {
        return;
      }

      // Bump the serial even on a cache hit so publishResult can't land a stale decode on top,
      // and raise the stale-fill floor: the cached frame IS the playhead now, so an abandoned
      // in-flight decode must not overwrite it one frame later (best-effort fills are for
      // starved playback, not for stepping the display backwards after a seek hit the cache).
      if (std::optional<DecodedCameraFrame> cached = cachedFrameLocked(key)) {
        stale_fill_min_serial_ = ++latest_serial_;
        queued_requests_.clear();
        pending_result_ = std::move(cached);
        active_key_.reset();
        failed_key_.reset();
        error_.clear();
        loading_ = false;
        return;
      }

      const uint64_t serial = ++latest_serial_;
      queued_requests_.clear();
      pending_result_.reset();
      queued_requests_.push_back(request_for_entry(index_, frame, serial, generation_, true));
      queuePrefetchLocked(frame_index, serial);
      active_key_ = key;
      failed_key_.reset();
      error_.clear();
      loading_ = true;
      queued = true;
      // Abort the in-flight decode only when it can't serve this target's segment (its reader is
      // obsolete). A same-segment decode is bounded work and gets to finish: during playback the
      // target advances every few frames, and killing the decode each time means no frame EVER
      // completes on a machine where a decode takes longer than one frame period — the canvas
      // just stays blank. The finished frame still reaches the screen via publishResult's
      // stale-serial path, then the fresher target decodes next.
      // Raised under mutex_: the worker consumes abort_ at dequeue under this same lock, so a
      // store after unlock could race past that reset and kill the NEW target's own decode.
      if (inflight_segment_.has_value() && *inflight_segment_ != frame.segment) abort_.store(true);
    }
    if (queued) cv_.notify_all();
  }

  std::optional<DecodedCameraFrame> take_frame() {
    std::optional<DecodedCameraFrame> frame;
    {
      std::lock_guard lock(mutex_);
      if (!pending_result_.has_value()) return std::nullopt;
      frame = std::move(pending_result_);
      pending_result_.reset();
      // A best-effort stale fill is NOT the active target: the real target's decode is still
      // queued/in-flight, so active_key_/loading_ must survive — clearing them re-queued the
      // same target from the keyframe on the next request_frame, doubling every decode.
      if (same_key(active_key_, frame->key)) {
        active_key_.reset();
        loading_ = false;
      }
      if (frame->ok) {
        displayed_key_ = frame->key;
        failed_key_.reset();
        error_.clear();
      } else {
        failed_key_ = frame->key;
        error_ = frame->error;
      }
    }
    return frame;
  }

  void invalidate_displayed() {
    std::lock_guard lock(mutex_);
    displayed_key_.reset();
  }

  CameraDecodeStatus status() const {
    std::lock_guard lock(mutex_);
    return CameraDecodeStatus{
      .has_source = !index_.segment_files.empty() && !index_.entries.empty(),
      .loading = loading_,
      .cached_frames = cache_.size(),
      .queued_frames = queued_requests_.size(),
      .active_key = active_key_,
      .displayed_key = displayed_key_,
      .failed_key = failed_key_,
      .error = error_,
    };
  }

private:
  std::optional<DecodedCameraFrame> cachedFrameLocked(const CameraDecodeKey &key) {
    for (auto it = cache_.begin(); it != cache_.end(); ++it) {
      if (!same_key(it->key, key)) continue;
      DecodedCameraFrame frame = *it;
      if (std::next(it) != cache_.end()) {
        DecodedCameraFrame cached = std::move(*it);
        cache_.erase(it);
        cache_.push_back(std::move(cached));
      }
      return frame;
    }
    return std::nullopt;
  }

  bool isCachedLocked(const CameraDecodeKey &key) const {
    return std::any_of(cache_.begin(), cache_.end(), [&](const DecodedCameraFrame &frame) {
      return frame.key == key;
    });
  }

  void queuePrefetchLocked(size_t frame_index, uint64_t serial) {
    constexpr size_t kPrefetchFrames = 2;
    for (size_t step = 1; step <= kPrefetchFrames; ++step) {
      const size_t next = frame_index + step;
      if (next >= index_.entries.size()) break;
      const CameraFrameIndexEntry &frame = index_.entries[next];
      const CameraDecodeKey key{.view = index_.view, .segment = frame.segment, .decode_index = frame.decode_index};
      if (isCachedLocked(key) || same_key(failed_key_, key)) continue;
      const bool already_queued = std::any_of(queued_requests_.begin(), queued_requests_.end(),
                                             [&](const DecodeRequest &request) {
        return request.key == key;
      });
      if (already_queued) continue;
      queued_requests_.push_back(request_for_entry(index_, frame, serial, generation_, false));
    }
  }

  void cacheFrame(const DecodeRequest &request, const DecodedCameraFrame &frame) {
    if (!frame.ok) return;
    std::lock_guard lock(mutex_);
    // Generation (not serial) gates caching: a decoded frame is valid LRU content even after its
    // target's serial is stale, which is what keeps lookahead prefetch durable across seeks.
    if (request.generation != generation_) return;
    cache_.erase(std::remove_if(cache_.begin(), cache_.end(), [&](const DecodedCameraFrame &cached) {
      return cached.key == frame.key;
    }), cache_.end());
    cache_.push_back(frame);
    while (cache_.size() > kMaxCachedFrames) cache_.pop_front();
  }

  std::shared_ptr<FrameReader> readerForRequest(const DecodeRequest &request) {
    std::lock_guard lock(reader_mutex_);
    if (reader_ != nullptr && reader_segment_ == request.key.segment && reader_generation_ == request.generation) {
      return reader_;
    }

    auto reader = std::make_shared<FrameReader>();
    if (!reader->load(decoder_camera_type(request.key.view), request.path, true, &abort_, true)) {
      return nullptr;
    }
    reader_ = reader;
    reader_segment_ = request.key.segment;
    reader_generation_ = request.generation;
    return reader_;
  }

  static bool ensureDecodeBuffer(FrameReader *reader, VisionBuf *buf, bool *allocated, int *width, int *height) {
    if (reader == nullptr || buf == nullptr || allocated == nullptr || width == nullptr || height == nullptr) {
      return false;
    }
    if (*allocated && *width == reader->width && *height == reader->height) return true;
    if (*allocated) {
      buf->free();
      *allocated = false;
    }
    const auto [stride, y_height, _uv_height, size] = get_nv12_info(reader->width, reader->height);
    buf->allocate(size);
    buf->init_yuv(reader->width, reader->height, stride, stride * y_height);
    *width = reader->width;
    *height = reader->height;
    *allocated = true;
    return true;
  }

  // Every worker path ends here exactly once per request (display or prefetch, ok or error), so
  // this is also where the in-flight marker read by request_frame's abort decision clears.
  void publishResult(const DecodeRequest &request, DecodedCameraFrame result) {
    std::lock_guard lock(mutex_);
    inflight_segment_.reset();
    if (!request.display) return;
    if (request.generation != generation_) return;
    // A fresh-serial result always wins the pending slot. A stale one (the tracker already moved
    // on) may still fill an EMPTY slot when it decoded ok: on a machine where every decode loses
    // the race against playback, dropping stale results means the canvas stays blank for the
    // whole run — a slightly-behind frame beats no frame, and the next fresh decode replaces it.
    if (request.serial != latest_serial_ &&
        (!result.ok || pending_result_.has_value() || request.serial < stale_fill_min_serial_)) {
      return;
    }
    pending_result_ = std::move(result);
  }

  void workerLoop() {
    VisionBuf decode_buffer;
    bool buffer_allocated = false;
    int buffer_width = 0;
    int buffer_height = 0;

    while (true) {
      DecodeRequest request;
      {
        std::unique_lock lock(mutex_);
        cv_.wait(lock, [&]() {
          return stop_.load() ||
                 !queued_requests_.empty();
        });
        if (stop_.load()) break;
        request = queued_requests_.front();
        queued_requests_.pop_front();
        inflight_segment_ = request.key.segment;
        // Sole point where abort_ is consumed, under the same lock request_frame and
        // set_camera_index hold when they raise it — any abort raised after this dequeue is
        // meant for THIS request and stays visible to both reader->load and reader->get. The
        // old per-stage resets (readerForRequest, pre-get) silently swallowed a cross-segment
        // abort that landed between dequeue and reset, leaving a seek stuck behind a full
        // stale-segment decode.
        abort_.store(false);
      }

      DecodedCameraFrame result;
      result.key = request.key;
      result.frame_id = request.frame_id;
      result.timestamp = request.timestamp;

      std::shared_ptr<FrameReader> reader = readerForRequest(request);
      if (reader == nullptr) {
        result.error = "failed to load camera segment";
        publishResult(request, std::move(result));
        continue;
      }

      if (!ensureDecodeBuffer(reader.get(), &decode_buffer, &buffer_allocated, &buffer_width, &buffer_height) ||
          !reader->get(request.key.decode_index, &decode_buffer)) {
        result.error = "failed to decode camera frame";
        publishResult(request, std::move(result));
        continue;
      }

      result.width = reader->width;
      result.height = reader->height;
      result.rgba.resize(static_cast<size_t>(result.width) * static_cast<size_t>(result.height) * 4U);
      libyuv::NV12ToABGR(decode_buffer.y,
                         static_cast<int>(decode_buffer.stride),
                         decode_buffer.uv,
                         static_cast<int>(decode_buffer.stride),
                         result.rgba.data(),
                         result.width * 4,
                         result.width,
                         result.height);
      result.ok = true;
      cacheFrame(request, result);
      publishResult(request, std::move(result));
    }

    if (buffer_allocated) decode_buffer.free();
  }

  // Protects decode queue/results/cache and the generation counters consumed by worker_.
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::thread worker_;
  // stop_ ends worker_; abort_ interrupts FrameReader when a newer request generation replaces the old one.
  std::atomic<bool> stop_{false};
  std::atomic<bool> abort_{false};
  CameraFeedIndex index_;
  uint64_t generation_ = 1;
  uint64_t latest_serial_ = 0;
  // Serial floor for publishResult's best-effort stale fill; bumped when a cache hit lands the
  // playhead frame so an older abandoned decode can't overwrite it a frame later.
  uint64_t stale_fill_min_serial_ = 0;
  bool loading_ = false;
  std::deque<DecodeRequest> queued_requests_;
  // Segment the worker is currently decoding; request_frame only aborts when this can't serve
  // the new target (see the comment there).
  std::optional<int> inflight_segment_;
  std::optional<DecodedCameraFrame> pending_result_;
  std::deque<DecodedCameraFrame> cache_;
  std::optional<CameraDecodeKey> active_key_;
  std::optional<CameraDecodeKey> displayed_key_;
  std::optional<CameraDecodeKey> failed_key_;
  std::string error_;

  // FrameReader is segment-scoped and reused across requests, so it has a separate short-held lock.
  std::mutex reader_mutex_;
  std::shared_ptr<FrameReader> reader_;
  int reader_segment_ = -1;
  uint64_t reader_generation_ = 0;

  static constexpr size_t kMaxCachedFrames = 4;
};

CameraFrameDecoder::CameraFrameDecoder() : impl_(std::make_unique<Impl>()) {}

CameraFrameDecoder::~CameraFrameDecoder() = default;

void CameraFrameDecoder::set_camera_index(CameraFeedIndex index) {
  impl_->set_camera_index(std::move(index));
}

void CameraFrameDecoder::update_camera_index(CameraFeedIndex index) {
  impl_->update_camera_index(std::move(index));
}

void CameraFrameDecoder::request_frame(double tracker_time) {
  impl_->request_frame(tracker_time);
}

std::optional<DecodedCameraFrame> CameraFrameDecoder::take_frame() {
  return impl_->take_frame();
}

void CameraFrameDecoder::invalidate_displayed() {
  impl_->invalidate_displayed();
}

CameraDecodeStatus CameraFrameDecoder::status() const {
  return impl_->status();
}

class LiveCameraWorker {
public:
  LiveCameraWorker(CameraViewKind view, std::string server_name)
      : view_(view), server_name_(std::move(server_name)), thread_([this]() { workerLoop(); }) {
    std::lock_guard lock(mutex_);
    status_.supported = true;
    status_.requested = true;
    status_.error = "waiting for " + server_name_ + " VisionIPC";
  }

  ~LiveCameraWorker() {
    stop();
  }

  LiveCameraWorker(const LiveCameraWorker &) = delete;
  LiveCameraWorker &operator=(const LiveCameraWorker &) = delete;

  void stop() {
    stop_.store(true);
    if (thread_.joinable()) thread_.join();
  }

  std::optional<DecodedCameraFrame> take_frame() {
    std::lock_guard lock(mutex_);
    if (!pending_frame_.has_value()) return std::nullopt;
    std::optional<DecodedCameraFrame> frame = std::move(pending_frame_);
    pending_frame_.reset();
    return frame;
  }

  LiveCameraFrameStatus status() const {
    std::lock_guard lock(mutex_);
    return status_;
  }

private:
  void setWaiting(std::string error, bool connected = false) {
    std::lock_guard lock(mutex_);
    status_.requested = true;
    status_.connected = connected;
    status_.error = std::move(error);
  }

  void setConnected(const VisionIpcClient &client) {
    const VisionBuf &buf = client.buffers[0];
    std::lock_guard lock(mutex_);
    status_.requested = true;
    status_.connected = true;
    status_.width = static_cast<int>(buf.width);
    status_.height = static_cast<int>(buf.height);
    status_.error.clear();
  }

  void publishFrame(DecodedCameraFrame frame) {
    std::lock_guard lock(mutex_);
    status_.requested = true;
    status_.connected = true;
    status_.has_frame = true;
    status_.received_frames += 1;
    status_.width = frame.width;
    status_.height = frame.height;
    status_.frame_id = frame.frame_id;
    status_.error.clear();
    pending_frame_ = std::move(frame);
  }

  bool convertFrame(VisionBuf *buf, const VisionIpcBufExtra &extra, DecodedCameraFrame *out) const {
    if (out == nullptr || buf == nullptr || buf->y == nullptr || buf->uv == nullptr ||
        buf->width == 0 || buf->height == 0 || buf->stride == 0) {
      return false;
    }

    const uint64_t raw_frame_id = extra.valid ? extra.frame_id : buf->get_frame_id();
    if (extra.valid && buf->get_frame_id() != raw_frame_id) {
      return false;
    }

    DecodedCameraFrame frame;
    frame.key = CameraDecodeKey{
      .view = view_,
      .segment = -1,
      .decode_index = clamp_decode_index(raw_frame_id),
    };
    frame.frame_id = clamp_frame_id(raw_frame_id);
    frame.timestamp = extra.timestamp_eof > 0 ? static_cast<double>(extra.timestamp_eof) / 1.0e9 : 0.0;
    frame.width = static_cast<int>(buf->width);
    frame.height = static_cast<int>(buf->height);
    frame.rgba.resize(static_cast<size_t>(frame.width) * static_cast<size_t>(frame.height) * 4U);
    libyuv::NV12ToABGR(buf->y,
                       static_cast<int>(buf->stride),
                       buf->uv,
                       static_cast<int>(buf->stride),
                       frame.rgba.data(),
                       frame.width * 4,
                       frame.width,
                       frame.height);
    frame.ok = true;
    *out = std::move(frame);
    return true;
  }

  void workerLoop() {
    const std::optional<VisionStreamType> stream_type = live_camera_stream_type(view_);
    if (!stream_type.has_value()) {
      setWaiting(live_camera_stream_label(view_) + " live VisionIPC stream is unavailable");
      return;
    }

    std::unique_ptr<VisionIpcClient> client;
    VisionIpcBufExtra extra = {};
    const std::string stream_label = live_camera_stream_label(view_);

    while (!stop_.load()) {
      if (client == nullptr || !client->connected) {
        const std::set<VisionStreamType> streams = VisionIpcClient::getAvailableStreams(server_name_, false);
        if (stop_.load()) break;
        if (streams.empty()) {
          setWaiting("waiting for " + server_name_ + " VisionIPC");
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
          continue;
        }
        if (streams.count(*stream_type) == 0) {
          setWaiting(stream_label + " stream is not available from " + server_name_);
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
          continue;
        }

        client = std::make_unique<VisionIpcClient>(server_name_, *stream_type, true);
        if (!client->connect(false) || client->num_buffers <= 0) {
          client.reset();
          setWaiting("connecting to " + stream_label + " VisionIPC");
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
          continue;
        }
        setConnected(*client);
      }

      VisionBuf *buf = client->recv(&extra, 100);
      if (stop_.load()) break;
      if (buf == nullptr) {
        if (!client->connected) {
          client.reset();
          setWaiting("reconnecting to " + stream_label + " VisionIPC");
        }
        continue;
      }

      DecodedCameraFrame frame;
      if (!convertFrame(buf, extra, &frame)) {
        setWaiting("dropped stale " + stream_label + " VisionIPC frame", true);
        continue;
      }
      publishFrame(std::move(frame));
    }
  }

  CameraViewKind view_ = CameraViewKind::Road;
  std::string server_name_;
  std::thread thread_;
  // Read by the VisionIPC worker loop and set by stop() before joining thread_.
  std::atomic<bool> stop_{false};
  // Protects status_ and the latest pending frame handed back to the UI thread.
  mutable std::mutex mutex_;
  std::optional<DecodedCameraFrame> pending_frame_;
  LiveCameraFrameStatus status_;
};

class LiveCameraFrameSource::Impl {
public:
  explicit Impl(std::string server_name) : server_name_(std::move(server_name)) {}

  ~Impl() {
    set_enabled(false);
  }

  void set_enabled(bool enabled) {
    std::array<std::shared_ptr<LiveCameraWorker>, 4> workers_to_stop;
    {
      std::lock_guard lock(mutex_);
      if (enabled_ == enabled && enabled) return;
      enabled_ = enabled;
      unsupported_requested_.fill(false);
      if (!enabled) {
        workers_to_stop = workers_;
        workers_.fill(nullptr);
      }
    }
    for (const std::shared_ptr<LiveCameraWorker> &worker : workers_to_stop) {
      if (worker != nullptr) worker->stop();
    }
  }

  void request_frame(CameraViewKind view) {
    std::lock_guard lock(mutex_);
    const size_t index = camera_view_index(view);
    if (!enabled_) return;
    if (!live_camera_view_supported(view)) {
      unsupported_requested_[index] = true;
      return;
    }
    if (workers_[index] == nullptr) {
      workers_[index] = std::make_shared<LiveCameraWorker>(view, server_name_);
    }
  }

  std::optional<DecodedCameraFrame> take_frame(CameraViewKind view) {
    std::shared_ptr<LiveCameraWorker> worker;
    {
      std::lock_guard lock(mutex_);
      worker = workers_[camera_view_index(view)];
    }
    return worker == nullptr ? std::nullopt : worker->take_frame();
  }

  LiveCameraFrameStatus status(CameraViewKind view) const {
    std::shared_ptr<LiveCameraWorker> worker;
    bool requested = false;
    bool enabled = false;
    {
      std::lock_guard lock(mutex_);
      const size_t index = camera_view_index(view);
      worker = workers_[index];
      requested = unsupported_requested_[index];
      enabled = enabled_;
    }
    if (worker != nullptr) return worker->status();

    LiveCameraFrameStatus status;
    status.supported = live_camera_view_supported(view);
    status.requested = requested;
    if (!status.supported && requested) {
      status.error = live_camera_stream_label(view) + " live VisionIPC stream is unavailable";
    } else if (enabled && requested) {
      status.error = "waiting for " + server_name_ + " VisionIPC";
    }
    return status;
  }

private:
  // Protects worker lifetime and unsupported-view request flags.
  mutable std::mutex mutex_;
  std::string server_name_ = "camerad";
  bool enabled_ = false;
  std::array<std::shared_ptr<LiveCameraWorker>, 4> workers_;
  std::array<bool, 4> unsupported_requested_{};
};

LiveCameraFrameSource::LiveCameraFrameSource(std::string server_name)
    : impl_(std::make_unique<Impl>(std::move(server_name))) {}

LiveCameraFrameSource::~LiveCameraFrameSource() = default;

void LiveCameraFrameSource::set_enabled(bool enabled) {
  impl_->set_enabled(enabled);
}

void LiveCameraFrameSource::request_frame(CameraViewKind view) {
  impl_->request_frame(view);
}

std::optional<DecodedCameraFrame> LiveCameraFrameSource::take_frame(CameraViewKind view) {
  return impl_->take_frame(view);
}

LiveCameraFrameStatus LiveCameraFrameSource::status(CameraViewKind view) const {
  return impl_->status(view);
}

}  // namespace loggy
