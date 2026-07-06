#include "tools/loggy/backend/live.h"

#include "tools/loggy/backend/panda_live.h"

#include "json11/json11.hpp"
#include "msgq/ipc.h"
#include "openpilot/cereal/messaging/bridge_zmq.h"
#include "openpilot/cereal/services.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iterator>
#include <memory>
#include <signal.h>
#include <stdexcept>
#include <thread>
#include <unordered_map>

#include <kj/exception.h>

#ifdef __linux__
#include <linux/can.h>
#include <linux/can/raw.h>
#include <net/if.h>
#include <sys/prctl.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace loggy {
namespace {

void extend_range(TimeRange *range, TimeRange incoming) {
  if (range == nullptr || !incoming.valid()) return;
  if (!range->valid()) {
    *range = incoming;
  } else {
    range->start_ = std::min(range->start_, incoming.start_);
    range->end = std::max(range->end, incoming.end);
  }
}

TimeRange live_batch_range(const LiveExtractBatch &batch) {
  TimeRange range;
  for (const TimeRange &coverage : batch.store.coverage) extend_range(&range, coverage);
  for (const SeriesChunk &chunk : batch.store.series) extend_range(&range, chunk.range);
  for (const CanEventChunk &chunk : batch.store.can_events) extend_range(&range, chunk.range);
  for (const TimelineSpan &span : batch.timeline_spans) extend_range(&range, {span.start_time, span.end_time});
  for (const LogEntry &entry : batch.logs) extend_range(&range, {entry.mono_time, entry.mono_time});
  return range;
}

double steady_seconds() {
  using Clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

std::filesystem::path messaging_bridge_path() {
  return loggy_repo_root_path() / "openpilot" / "cereal" / "messaging" / "bridge";
}

#ifdef __linux__
class DeviceBridgeProcess {
public:
  DeviceBridgeProcess() = default;
  DeviceBridgeProcess(const DeviceBridgeProcess &) = delete;
  DeviceBridgeProcess &operator=(const DeviceBridgeProcess &) = delete;
  ~DeviceBridgeProcess() { stop(); }

  void start(const std::string &address) {
    const std::filesystem::path bridge = messaging_bridge_path();
    if (!std::filesystem::exists(bridge)) {
      throw std::runtime_error("messaging bridge binary not found: " + bridge.string());
    }
    const pid_t parent_pid = getpid();
    pid_ = fork();
    if (pid_ == 0) {
      prctl(PR_SET_PDEATHSIG, SIGKILL);
      if (getppid() != parent_pid) _exit(0);
      const std::string bridge_path = bridge.string();
      execl(bridge_path.c_str(), bridge_path.c_str(), address.c_str(), "/\"can/\"", static_cast<char *>(nullptr));
      _exit(127);
    }
    if (pid_ < 0) {
      pid_ = -1;
      throw std::runtime_error("Failed to start messaging bridge");
    }
  }

  void stop() {
    if (pid_ <= 0) return;
    kill(pid_, SIGKILL);
    int status = 0;
    waitpid(pid_, &status, 0);
    pid_ = -1;
  }

private:
  pid_t pid_ = -1;
};
#endif

void merge_store_batch(StoreBatch *dst, StoreBatch *src) {
  if (dst == nullptr || src == nullptr) return;
  dst->coverage.insert(dst->coverage.end(), src->coverage.begin(), src->coverage.end());
  dst->series.insert(dst->series.end(),
                     std::make_move_iterator(src->series.begin()),
                     std::make_move_iterator(src->series.end()));
  dst->can_events.insert(dst->can_events.end(),
                         std::make_move_iterator(src->can_events.begin()),
                         std::make_move_iterator(src->can_events.end()));
}

void merge_live_batch(LiveExtractBatch *dst, LiveExtractBatch *src) {
  if (dst == nullptr || src == nullptr) return;
  if (src->has_time_offset) {
    dst->has_time_offset = true;
    dst->time_offset = src->time_offset;
  }
  merge_store_batch(&dst->store, &src->store);
  dst->timeline_spans.insert(dst->timeline_spans.end(),
                             std::make_move_iterator(src->timeline_spans.begin()),
                             std::make_move_iterator(src->timeline_spans.end()));
  dst->logs.insert(dst->logs.end(),
                   std::make_move_iterator(src->logs.begin()),
                   std::make_move_iterator(src->logs.end()));
  if (dst->car_fingerprint.empty() && !src->car_fingerprint.empty()) {
    dst->car_fingerprint = std::move(src->car_fingerprint);
  }
  extend_range(&dst->range, src->range);
  dst->events_seen += src->events_seen;
  dst->events_appended += src->events_appended;
}

template <typename Msg>
bool append_message_payload(Msg *msg, LiveCerealAccumulator *accumulator) {
  if (msg == nullptr || accumulator == nullptr) return false;
  const size_t size = msg->getSize();
  if (size < sizeof(capnp::word) || (size % sizeof(capnp::word)) != 0) return false;
  const size_t word_count = size / sizeof(capnp::word);
  const auto *raw = reinterpret_cast<const unsigned char *>(msg->getData());
  if ((reinterpret_cast<uintptr_t>(raw) % alignof(capnp::word)) == 0) {
    kj::ArrayPtr<const capnp::word> data(reinterpret_cast<const capnp::word *>(raw), word_count);
    return accumulator->append_serialized(data);
  }

  kj::Array<capnp::word> aligned = kj::heapArray<capnp::word>(word_count);
  std::memcpy(aligned.begin(), raw, size);
  return accumulator->append_serialized(aligned.asPtr());
}

}  // namespace

LiveCerealPoller::LiveCerealPoller() = default;

LiveCerealPoller::~LiveCerealPoller() { stop(); }

bool live_is_local_stream_address(std::string_view address) {
  return address.empty() || address == "127.0.0.1" || address == "localhost";
}

std::string normalize_live_stream_address(std::string address) {
  return live_is_local_stream_address(address) ? "127.0.0.1" : address;
}

const char *live_source_kind_label(LiveSourceKind kind) {
  switch (kind) {
    case LiveSourceKind::SocketCan: return "SocketCAN";
    case LiveSourceKind::PandaUsb: return "Panda USB";
    case LiveSourceKind::DeviceBridge: return "Device Bridge";
    case LiveSourceKind::CerealRemote: return "Remote (ZMQ)";
    case LiveSourceKind::CerealLocal:
    default: return "Local (MSGQ)";
  }
}

std::string live_source_target_label(const LiveSourceConfig &source) {
  if (source.kind == LiveSourceKind::SocketCan) return source.address.empty() ? "socketcan" : source.address;
  if (source.kind == LiveSourceKind::PandaUsb) return source.address.empty() ? "first Panda" : source.address;
  if (source.kind == LiveSourceKind::DeviceBridge) return source.address.empty() ? "127.0.0.1" : source.address;
  return source.kind == LiveSourceKind::CerealRemote ? normalize_live_stream_address(source.address) : "127.0.0.1";
}

bool live_panda_can_speed_supported(uint16_t speed_kbps) {
  return std::find(kPandaCanSpeedsKbps.begin(), kPandaCanSpeedsKbps.end(), speed_kbps) != kPandaCanSpeedsKbps.end();
}

bool live_panda_data_speed_supported(uint16_t speed_kbps) {
  return std::find(kPandaDataSpeedsKbps.begin(), kPandaDataSpeedsKbps.end(), speed_kbps) != kPandaDataSpeedsKbps.end();
}

PandaBusConfig normalize_live_panda_bus_config(PandaBusConfig config) {
  if (!live_panda_can_speed_supported(config.can_speed_kbps)) config.can_speed_kbps = PandaBusConfig{}.can_speed_kbps;
  if (!live_panda_data_speed_supported(config.data_speed_kbps)) config.data_speed_kbps = PandaBusConfig{}.data_speed_kbps;
  return config;
}

bool live_should_subscribe_service(std::string_view name) {
  static const std::array<std::string_view, 13> skipped = {{
    "roadEncodeIdx",
    "driverEncodeIdx",
    "wideRoadEncodeIdx",
    "qRoadEncodeIdx",
    "roadEncodeData",
    "driverEncodeData",
    "wideRoadEncodeData",
    "qRoadEncodeData",
    "livestreamWideRoadEncodeIdx",
    "livestreamRoadEncodeIdx",
    "livestreamDriverEncodeIdx",
    "thumbnail",
  }};
  if (name == "rawAudioData") return false;
  return std::find(skipped.begin(), skipped.end(), name) == skipped.end();
}

bool live_batch_has_data(const LiveExtractBatch &batch) {
  return !batch.store.series.empty()
      || !batch.store.can_events.empty()
      || !batch.timeline_spans.empty()
      || !batch.logs.empty();
}

std::vector<std::string> live_panda_serials() {
  try {
    return panda_live_serials();
  } catch (const std::exception &) {
    return {};
  }
}

bool live_panda_available() {
  return !live_panda_serials().empty();
}

bool live_socketcan_available() {
#ifdef __linux__
  const int fd = socket(PF_CAN, SOCK_RAW, CAN_RAW);
  if (fd < 0) return false;
  ::close(fd);
  return true;
#else
  return false;
#endif
}

LiveExtractBatch make_live_can_batch(const MessageId &id, std::vector<uint8_t> data,
                                     double mono_time, uint16_t bus_time) {
  LiveExtractBatch batch;
  batch.store.segment = -1;
  batch.store.coverage.push_back({mono_time, mono_time});
  batch.store.can_events.push_back({
    .id = id,
    .range = {mono_time, mono_time},
    .events = {{.mono_time = mono_time, .bus_time = bus_time, .data = std::move(data)}},
    .segment = -1,
  });
  batch.range = {mono_time, mono_time};
  batch.events_seen = 1;
  batch.events_appended = 1;
  return batch;
}

LiveCerealAccumulator::LiveCerealAccumulator(std::optional<double> time_offset)
  : time_offset_(time_offset) {
  options_.segment = -1;
  options_.include_raw_can_events = true;
  options_.time_offset = time_offset_;
  reset_series();
}

void LiveCerealAccumulator::reset_series() {
  series_ = make_series_accumulator(event_schema_index(), -1);
}

bool LiveCerealAccumulator::append_serialized(kj::ArrayPtr<const capnp::word> data) {
  try {
    capnp::ReaderOptions options;
    options.traversalLimitInWords = kj::maxValue;
    capnp::FlatArrayMessageReader reader(data, options);
    return append_event(reader.getRoot<cereal::Event>());
  } catch (const kj::Exception &) {
    return false;
  }
}

bool LiveCerealAccumulator::append_event(const cereal::Event::Reader &event) {
  // Increment counters in this parser-only accumulator before any reset in take_batch().
  ++events_seen_;
  const double boot_time = static_cast<double>(event.getLogMonoTime()) / 1.0e9;
  if (!time_offset_.has_value()) {
    time_offset_ = boot_time;
    options_.time_offset = time_offset_;
  }
  const double offset = time_offset_.value_or(0.0);
  const cereal::Event::Which which = event.which();
  const bool appended = append_event_reader(which, event, options_, &series_);
  if (appended) ++events_appended_;

  if (which == cereal::Event::Which::SELFDRIVE_STATE) {
    const auto selfdrive = event.getSelfdriveState();
    append_timeline_point(&timeline_spans_, boot_time - offset,
                          timeline_kind_for_selfdrive(selfdrive.getAlertStatus(), selfdrive.getEnabled()));
  } else if (which == cereal::Event::Which::CAR_PARAMS) {
    const std::string fingerprint = event.getCarParams().getCarFingerprint().cStr();
    if (!fingerprint.empty()) car_fingerprint_ = fingerprint;
  }
  append_log_event(which, event, offset, &logs_, last_alert_key_);
  return appended;
}

LiveExtractBatch LiveCerealAccumulator::take_batch() {
  LiveExtractBatch batch;
  SegmentExtractResult extracted = series_.finish({});
  batch.store = std::move(extracted.batch);
  batch.timeline_spans.swap(timeline_spans_);
  batch.logs.swap(logs_);
  batch.car_fingerprint = car_fingerprint_;
  batch.has_time_offset = time_offset_.has_value();
  batch.time_offset = time_offset_.value_or(0.0);
  batch.events_seen = events_seen_;
  batch.events_appended = events_appended_;
  batch.range = live_batch_range(batch);
  events_seen_ = 0;
  events_appended_ = 0;
  reset_series();
  return batch;
}

void LiveCerealPoller::start(LiveSourceConfig requested_source, std::optional<double> time_offset) {
  stop();
  {
    // Reinitialize shared state under mutex before starting the worker so snapshot/consume observe a clean frame.
    std::lock_guard lock(mutex_);
    pending_ = {};
    error_.clear();
    source_ = std::move(requested_source);
    source_.buffer_seconds = std::max(1.0, source_.buffer_seconds);
    if (source_.kind == LiveSourceKind::CerealLocal) {
      source_.address = "127.0.0.1";
    } else if (source_.kind == LiveSourceKind::CerealRemote || source_.kind == LiveSourceKind::DeviceBridge) {
      source_.address = normalize_live_stream_address(source_.address);
    }
  }
  received_messages_.store(0);
  parsed_messages_.store(0);
  dropped_messages_.store(0);
  published_batches_.store(0);
  connected_.store(false);
  paused_.store(false);
  running_.store(true);
  worker_ = std::thread([this, time_offset]() {
      try {
        LiveSourceConfig source;
        {
        // Capture source once at thread start; workers must not race with live UI mutations mid-flight.
          std::lock_guard lock(mutex_);
          source = source_;
        }
      if (source.kind == LiveSourceKind::SocketCan) {
        run_socket_can_source(source);
      } else if (source.kind == LiveSourceKind::PandaUsb) {
        run_panda_source(source);
      } else {
        LiveCerealAccumulator accumulator(time_offset);
        if (source.kind == LiveSourceKind::DeviceBridge) {
          run_device_bridge_source(source, &accumulator);
        } else {
          run_cereal_source(source, &accumulator);
        }
      }
    } catch (const std::exception &err) {
      std::lock_guard lock(mutex_);
      error_ = err.what();
    }
    // Worker teardown is the only place that flips both lifecycle flags false together.
    connected_.store(false);
    running_.store(false);
  });
}

void LiveCerealPoller::set_paused(bool paused) {
  // Pause is atomic for loop checks; pending/error are still mutex-protected.
  paused_.store(paused);
  if (paused) {
    std::lock_guard lock(mutex_);
    pending_ = {};
    error_.clear();
  }
}

void LiveCerealPoller::stop() {
  // stop() drives the worker loop via atomic state before waiting to join thread lifetime.
  running_.store(false);
  paused_.store(false);
  if (worker_.joinable()) worker_.join();
  connected_.store(false);
}

LivePollSnapshot LiveCerealPoller::snapshot() const {
  LivePollSnapshot out;
  // Sample atomics first, then lock source/error so callers can read lock-free counters with coherent source metadata.
  out.active = running_.load();
  out.connected = connected_.load();
  out.paused = paused_.load();
  out.received_messages = received_messages_.load();
  out.parsed_messages = parsed_messages_.load();
  out.dropped_messages = dropped_messages_.load();
  out.published_batches = published_batches_.load();
  std::lock_guard lock(mutex_);
  out.source_kind = source_.kind;
  out.source_label = live_source_target_label(source_);
  out.buffer_seconds = source_.buffer_seconds;
  out.error = error_;
  return out;
}

LivePollResult LiveCerealPoller::consume(LiveExtractBatch &batch) {
  // consume takes mutex to move pending_ out as one atomic operation with error clearing.
  std::lock_guard lock(mutex_);
  const bool has_error = !error_.empty();
  const bool has_batch = live_batch_has_data(pending_);
  if (!has_error && !has_batch) return {};

  LivePollResult result;
  result.has_update = true;
  result.error = std::move(error_);
  error_.clear();
  if (has_batch) {
    batch = std::move(pending_);
    pending_ = {};
    result.has_batch = true;
  }
  return result;
}

void LiveCerealPoller::publish_batch(LiveCerealAccumulator *accumulator) {
  if (accumulator == nullptr) return;
  LiveExtractBatch batch = accumulator->take_batch();
  if (!live_batch_has_data(batch)) return;
  // Merge into pending_ under mutex to keep pending/error ownership single-producer/single-consumer.
  std::lock_guard lock(mutex_);
  merge_live_batch(&pending_, &batch);
  published_batches_.fetch_add(1);
}

void LiveCerealPoller::run_cereal_source(const LiveSourceConfig &source, LiveCerealAccumulator *accumulator) {
  if (source.kind == LiveSourceKind::CerealRemote) {
    run_remote_cereal_source(source, accumulator);
    return;
  }
  run_local_cereal_source(source, accumulator);
}

void LiveCerealPoller::run_device_bridge_source(const LiveSourceConfig &source, LiveCerealAccumulator *accumulator) {
#ifdef __linux__
    DeviceBridgeProcess bridge;
    bridge.start(source.address.empty() ? "127.0.0.1" : source.address);
    LiveSourceConfig local = source;
    local.kind = LiveSourceKind::CerealLocal;
    local.address = "127.0.0.1";
    run_local_cereal_source(local, accumulator);
#else
    (void)source;
    (void)accumulator;
    throw std::runtime_error("Device bridge is only available on Linux");
#endif
}

void LiveCerealPoller::publish_can_batch(LiveExtractBatch batch) {
  if (!live_batch_has_data(batch)) return;
  // Keep publish path lock-scoped even for CAN sources, so consume() never sees a partial batch.
  std::lock_guard lock(mutex_);
  merge_live_batch(&pending_, &batch);
  published_batches_.fetch_add(1);
}

void LiveCerealPoller::run_panda_source(const LiveSourceConfig &source) {
#ifdef __linux__
    std::unique_ptr<PandaLiveReader> panda;
    const std::string serial = source.address;
    const auto connect_panda = [&]() {
      connected_.store(false);
      panda = std::make_unique<PandaLiveReader>(serial, source.panda_buses);
      if (!panda->connected()) throw std::runtime_error("Panda USB connection is unhealthy");
      connected_.store(true);
    };

    try {
      connect_panda();
    } catch (const std::exception &err) {
      throw std::runtime_error((serial.empty()
        ? "Failed to connect to Panda USB"
        : "Failed to connect to Panda USB " + serial) + std::string(": ") + err.what());
    }

  const double start_ = steady_seconds();
  uint16_t bus_time = 0;
  std::vector<PandaLiveFrame> raw_frames;
  while (running_.load()) {
    // running_ is the long-lived cancellation flag checked at each loop iteration.
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
      if (panda == nullptr || !panda->connected()) {
        try {
          connect_panda();
        } catch (const std::exception &) {
          std::this_thread::sleep_for(std::chrono::seconds(1));
          continue;
        }
      }

      raw_frames.clear();
      if (!panda->receive(&raw_frames)) {
        dropped_messages_.fetch_add(1);
        continue;
      }
      if (raw_frames.empty()) {
        panda->send_heartbeat(false);
        continue;
      }
      received_messages_.fetch_add(raw_frames.size());
      if (!paused_.load()) {
        LiveExtractBatch batch;
        const double mono_time = std::max(0.0, steady_seconds() - start_);
        for (PandaLiveFrame &frame : raw_frames) {
          LiveExtractBatch frame_batch = make_live_can_batch(
            MessageId{.source = frame.source, .address = frame.address},
            std::move(frame.data), mono_time, bus_time++);
          merge_live_batch(&batch, &frame_batch);
          parsed_messages_.fetch_add(1);
        }
        publish_can_batch(std::move(batch));
      }
      panda->send_heartbeat(false);
    }
#else
    (void)source;
    throw std::runtime_error("Panda USB is only available on Linux");
#endif
}

void LiveCerealPoller::run_socket_can_source(const LiveSourceConfig &source) {
#ifdef __linux__
    if (!live_socketcan_available()) throw std::runtime_error("SocketCAN not available");
    if (source.address.empty()) throw std::runtime_error("SocketCAN device is empty");

    const int fd = socket(PF_CAN, SOCK_RAW, CAN_RAW);
    if (fd < 0) throw std::runtime_error("Failed to create SocketCAN socket");

    const auto close_fd = [&]() {
      if (fd >= 0) ::close(fd);
    };

    int enable_fd = 1;
    setsockopt(fd, SOL_CAN_RAW, CAN_RAW_FD_FRAMES, &enable_fd, sizeof(enable_fd));

    struct ifreq ifr = {};
    std::strncpy(ifr.ifr_name, source.address.c_str(), IFNAMSIZ - 1);
    if (ioctl(fd, SIOCGIFINDEX, &ifr) < 0) {
      close_fd();
      throw std::runtime_error("Failed to find SocketCAN device " + source.address);
    }

    struct sockaddr_can addr = {};
    addr.can_family = AF_CAN;
    addr.can_ifindex = ifr.ifr_ifindex;
    if (bind(fd, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)) < 0) {
      close_fd();
      throw std::runtime_error("Failed to bind SocketCAN device " + source.address);
    }

    struct timeval timeout = {.tv_sec = 0, .tv_usec = 100000};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
  connected_.store(true);

    const double start_ = steady_seconds();
    uint16_t bus_time = 0;
    while (running_.load()) {
      struct canfd_frame frame = {};
      const ssize_t nbytes = read(fd, &frame, sizeof(frame));
      if (nbytes <= 0) continue;
      if ((frame.can_id & CAN_ERR_FLAG) != 0) continue;
      received_messages_.fetch_add(1);
      if (paused_.load()) continue;

      const bool extended = (frame.can_id & CAN_EFF_FLAG) != 0;
      const uint32_t address = frame.can_id & (extended ? CAN_EFF_MASK : CAN_SFF_MASK);
      const size_t len = std::min<size_t>(frame.len, CAN_MAX_DATA_BYTES);
      std::vector<uint8_t> data(frame.data, frame.data + len);
      const double mono_time = std::max(0.0, steady_seconds() - start_);
      publish_can_batch(make_live_can_batch(MessageId{.source = 0, .address = address},
                                          std::move(data), mono_time, bus_time++));
      parsed_messages_.fetch_add(1);
    }
    close_fd();
#else
    (void)source;
    throw std::runtime_error("SocketCAN not available on this platform");
#endif
}

void LiveCerealPoller::run_local_cereal_source(const LiveSourceConfig &source, LiveCerealAccumulator *accumulator) {
    unsetenv("ZMQ");

    std::unique_ptr<Context> context(Context::create());
    std::unique_ptr<Poller> poller(Poller::create());
    std::vector<std::unique_ptr<SubSocket>> sockets;
    for (const auto &[name, info] : services) {
      if (!live_should_subscribe_service(name)) continue;
      std::unique_ptr<SubSocket> socket(
        SubSocket::create(context.get(), name.c_str(), source.address.c_str(), false, true, info.queue_size));
      if (socket == nullptr) continue;
      socket->setTimeout(0);
      poller->registerSocket(socket.get());
      sockets.push_back(std::move(socket));
    }
    if (sockets.empty()) throw std::runtime_error("Failed to connect to any cereal service");
    connected_.store(true);

    while (running_.load()) {
      std::vector<SubSocket *> ready = poller->poll(10);
      for (SubSocket *socket : ready) {
        while (running_.load()) {
          std::unique_ptr<Message> msg(socket->receive(true));
          if (!msg) break;
          received_messages_.fetch_add(1);
          if (paused_.load()) continue;
          if (append_message_payload(msg.get(), accumulator)) {
            parsed_messages_.fetch_add(1);
          } else {
            dropped_messages_.fetch_add(1);
          }
        }
      }
      publish_batch(accumulator);
    }
}

void LiveCerealPoller::run_remote_cereal_source(const LiveSourceConfig &source, LiveCerealAccumulator *accumulator) {
    BridgeZmqContext context;
    BridgeZmqPoller poller;
    std::vector<std::unique_ptr<BridgeZmqSubSocket>> sockets;
    for (const auto &[name, _] : services) {
      if (!live_should_subscribe_service(name)) continue;
      auto socket = std::make_unique<BridgeZmqSubSocket>();
      if (socket->connect(&context, name, source.address, false, true) != 0) continue;
      socket->setTimeout(0);
      poller.registerSocket(socket.get());
      sockets.push_back(std::move(socket));
    }
    if (sockets.empty()) throw std::runtime_error("Failed to connect to any remote cereal service");
    connected_.store(true);

    while (running_.load()) {
      std::vector<BridgeZmqSubSocket *> ready = poller.poll(10);
      for (BridgeZmqSubSocket *socket : ready) {
        while (running_.load()) {
          std::unique_ptr<BridgeZmqMessage> msg(static_cast<BridgeZmqMessage *>(socket->receive(true)));
          if (!msg) break;
          received_messages_.fetch_add(1);
          if (paused_.load()) continue;
          if (append_message_payload(static_cast<Message *>(msg.get()), accumulator)) {
            parsed_messages_.fetch_add(1);
          } else {
            dropped_messages_.fetch_add(1);
          }
        }
      }
      publish_batch(accumulator);
    }
}

}  // namespace loggy
