#pragma once

#include "tools/loggy/backend/extract.h"
#include "tools/loggy/backend/route.h"

#include <atomic>
#include <array>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#include <capnp/serialize.h>

namespace loggy {

enum class LiveSourceKind : uint8_t {
  CerealLocal,
  CerealRemote,
  DeviceBridge,
  PandaUsb,
  SocketCan,
};

inline constexpr size_t kPandaBusCount = 3;
inline constexpr std::array<uint16_t, 8> kPandaCanSpeedsKbps = {{10, 20, 50, 100, 125, 250, 500, 1000}};
inline constexpr std::array<uint16_t, 10> kPandaDataSpeedsKbps = {{10, 20, 50, 100, 125, 250, 500, 1000, 2000, 5000}};

struct PandaBusConfig {
  uint16_t can_speed_kbps = 500;
  uint16_t data_speed_kbps = 2000;
  bool can_fd = false;
};

struct LiveSourceConfig {
  LiveSourceKind kind = LiveSourceKind::CerealLocal;
  std::string address = "127.0.0.1";
  std::array<PandaBusConfig, kPandaBusCount> panda_buses{};
  double buffer_seconds = 30.0;
};

struct LiveExtractBatch {
  StoreBatch store;
  std::vector<TimelineSpan> timeline_spans;
  std::vector<LogEntry> logs;
  std::string car_fingerprint;
  bool has_time_offset = false;
  double time_offset = 0.0;
  TimeRange range;
  size_t events_seen = 0;
  size_t events_appended = 0;
};

struct LivePollSnapshot {
  bool active = false;
  bool connected = false;
  bool paused = false;
  LiveSourceKind source_kind = LiveSourceKind::CerealLocal;
  std::string source_label = "127.0.0.1";
  std::string error;
  double buffer_seconds = 30.0;
  uint64_t received_messages = 0;
  uint64_t parsed_messages = 0;
  uint64_t dropped_messages = 0;
  uint64_t published_batches = 0;
};

struct LivePollResult {
  bool has_update = false;
  bool has_batch = false;
  std::string error;
};

bool live_is_local_stream_address(std::string_view address);
std::string normalize_live_stream_address(std::string address);
const char *live_source_kind_label(LiveSourceKind kind);
std::string live_source_target_label(const LiveSourceConfig &source);
bool live_panda_can_speed_supported(uint16_t speed_kbps);
bool live_panda_data_speed_supported(uint16_t speed_kbps);
PandaBusConfig normalize_live_panda_bus_config(PandaBusConfig config);
bool live_should_subscribe_service(std::string_view name);
bool live_batch_has_data(const LiveExtractBatch &batch);
std::vector<std::string> live_panda_serials();
bool live_panda_available();
bool live_socketcan_available();
LiveExtractBatch make_live_can_batch(const MessageId &id, std::vector<uint8_t> data,
                                     double mono_time, uint16_t bus_time = 0);

class LiveCerealAccumulator {
public:
  explicit LiveCerealAccumulator(std::optional<double> time_offset = std::nullopt);

  bool append_serialized(kj::ArrayPtr<const capnp::word> data);
  bool append_event(const cereal::Event::Reader &event);
  LiveExtractBatch take_batch();

  std::optional<double> time_offset() const { return time_offset_; }
  size_t events_seen() const { return events_seen_; }
  size_t events_appended() const { return events_appended_; }

private:
  void reset_series();

  SeriesAccumulator series_;
  SegmentExtractOptions options_;
  std::optional<double> time_offset_;
  std::vector<TimelineSpan> timeline_spans_;
  std::vector<LogEntry> logs_;
  std::string car_fingerprint_;
  std::string last_alert_key_;
  size_t events_seen_ = 0;
  size_t events_appended_ = 0;
};

class LiveCerealPoller {
public:
  LiveCerealPoller();
  ~LiveCerealPoller();

  LiveCerealPoller(const LiveCerealPoller &) = delete;
  LiveCerealPoller &operator=(const LiveCerealPoller &) = delete;

  void start(LiveSourceConfig config, std::optional<double> time_offset = std::nullopt);
  void set_paused(bool paused);
  void stop();
  LivePollSnapshot snapshot() const;
  LivePollResult consume(LiveExtractBatch &batch);

private:
  void publish_batch(LiveCerealAccumulator *accumulator);
  void publish_can_batch(LiveExtractBatch batch);
  void run_cereal_source(const LiveSourceConfig &source, LiveCerealAccumulator *accumulator);
  void run_device_bridge_source(const LiveSourceConfig &source, LiveCerealAccumulator *accumulator);
  void run_panda_source(const LiveSourceConfig &source);
  void run_socket_can_source(const LiveSourceConfig &source);
  void run_local_cereal_source(const LiveSourceConfig &source, LiveCerealAccumulator *accumulator);
  void run_remote_cereal_source(const LiveSourceConfig &source, LiveCerealAccumulator *accumulator);

  mutable std::mutex mutex_;  // Guards pending_, error_, and source_ snapshots.
  std::thread worker_;
  std::atomic<bool> running_{false};   // Worker lifecycle flag read by the thread and control methods.
  std::atomic<bool> connected_{false};  // Published connection state for snapshot readers.
  std::atomic<bool> paused_{false};     // Pause gate checked by the worker loop.
  std::atomic<uint64_t> received_messages_{0};
  std::atomic<uint64_t> parsed_messages_{0};
  std::atomic<uint64_t> dropped_messages_{0};
  std::atomic<uint64_t> published_batches_{0};
  LiveSourceConfig source_;
  LiveExtractBatch pending_;
  std::string error_;
};

}  // namespace loggy
