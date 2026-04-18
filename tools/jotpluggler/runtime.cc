#include "tools/jotpluggler/app.h"
#include "tools/jotpluggler/common.h"

#include "cereal/services.h"
#include "common/timing.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_opengl3_loader.h"
#include "implot.h"
#include "libyuv.h"
#include "msgq_repo/msgq/ipc.h"
#include "tools/replay/framereader.h"

#include <GLFW/glfw3.h>

#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <memory>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <unistd.h>

#include "system/camerad/cameras/nv12_info.h"

namespace {

std::atomic<bool> g_glfw_alive{false};
const bool kLogCameraTimings = env_flag_enabled("JOTP_CAMERA_TIMINGS");

CameraType decoder_camera_type(CameraViewKind view) {
  switch (view) {
    case CameraViewKind::Driver: return DriverCam;
    case CameraViewKind::WideRoad: return WideRoadCam;
    case CameraViewKind::QRoad: return RoadCam;
    case CameraViewKind::Road:
    default: return RoadCam;
  }
}

bool stream_batch_has_data(const StreamExtractBatch &batch) {
  return !batch.series.empty()
      || !batch.can_messages.empty()
      || !batch.logs.empty()
      || !batch.timeline.empty()
      || !batch.enum_info.empty()
      || !batch.car_fingerprint.empty()
      || !batch.dbc_name.empty();
}

bool should_subscribe_stream_service(const std::string &name) {
  static const std::array<std::string_view, 13> kSkippedServices = {{
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
    "navThumbnail",
  }};
  if (name == "rawAudioData") return false;
  for (std::string_view skipped : kSkippedServices) {
    if (name == skipped) return false;
  }
  return true;
}

void glfw_error_callback(int error, const char *description) {
  const std::string_view desc = description != nullptr ? description : "unknown";
  if (error == 65539 && desc.find("Invalid window attribute 0x0002000D") != std::string_view::npos) {
    return;
  }
  std::cerr << "GLFW error " << error << ": " << desc << "\n";
}

}  // namespace

GlfwRuntime::GlfwRuntime(const Options &options) {
  glfwSetErrorCallback(glfw_error_callback);
  if (!glfwInit()) throw std::runtime_error("glfwInit failed");
  g_glfw_alive.store(true);

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
#endif
  const bool fixed_size = !options.show;
  glfwWindowHint(GLFW_RESIZABLE, fixed_size ? GLFW_FALSE : GLFW_TRUE);
  glfwWindowHint(GLFW_VISIBLE, options.show ? GLFW_TRUE : GLFW_FALSE);

  window_ = glfwCreateWindow(options.width, options.height, "jotpluggler", nullptr, nullptr);
  if (window_ == nullptr) {
    glfwTerminate();
    throw std::runtime_error("glfwCreateWindow failed");
  }

  if (fixed_size) {
    glfwSetWindowSizeLimits(window_, options.width, options.height, options.width, options.height);
  }
  glfwMakeContextCurrent(window_);
  glfwSwapInterval(options.show ? 1 : 0);
}

GlfwRuntime::~GlfwRuntime() {
  if (window_ != nullptr) {
    glfwDestroyWindow(window_);
  }
  g_glfw_alive.store(false);
  glfwTerminate();
}

GLFWwindow *GlfwRuntime::window() const {
  return window_;
}

ImGuiRuntime::ImGuiRuntime(GLFWwindow *window) {
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImPlot::CreateContext();

  ImGuiIO &io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
  io.IniFilename = nullptr;
  io.LogFilename = nullptr;

  if (!ImGui_ImplGlfw_InitForOpenGL(window, true)) {
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    throw std::runtime_error("ImGui_ImplGlfw_InitForOpenGL failed");
  }
  if (!ImGui_ImplOpenGL3_Init("#version 330")) {
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    throw std::runtime_error("ImGui_ImplOpenGL3_Init failed");
  }
}

ImGuiRuntime::~ImGuiRuntime() {
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImPlot::DestroyContext();
  ImGui::DestroyContext();
}

struct TerminalRouteProgress::Impl {
  explicit Impl(bool enabled) : enabled_(enabled) {}

  void update(const RouteLoadProgress &progress) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!enabled_) {
      return;
    }

    double overall = 0.0;
    std::string detail = "Resolving route";
    if (progress.stage == RouteLoadStage::Finished) {
      overall = 1.0;
      detail = "Ready";
    } else if (progress.total_segments > 0) {
      const bool finalizing = progress.segments_downloaded >= progress.total_segments
                           && progress.segments_parsed >= progress.total_segments;
      if (finalizing) {
        overall = 0.99;
        detail = "Finalizing route data";
      } else {
        const double total_work = static_cast<double>(progress.total_segments) * 2.0;
        const double complete_work = static_cast<double>(progress.segments_downloaded + progress.segments_parsed);
        overall = total_work <= 0.0 ? 0.0 : std::clamp(complete_work / total_work, 0.0, 0.99);
        std::ostringstream desc;
        desc << "Downloaded " << progress.segments_downloaded << "/" << progress.total_segments
             << "  Parsed " << progress.segments_parsed << "/" << progress.total_segments;
        detail = desc.str();
      }
    }

    render(overall, detail);
  }

  void finish() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!enabled_ || !rendered_) {
      return;
    }
    render(1.0, "Ready");
    std::fputc('\n', stderr);
    std::fflush(stderr);
    rendered_ = false;
  }

  void render(double progress, const std::string &detail) {
    const int percent = std::clamp(static_cast<int>(std::round(progress * 100.0)), 0, 100);
    if (percent == last_percent_ && detail == last_detail_) {
      return;
    }

    constexpr int kWidth = 20;
    const int filled = std::clamp(static_cast<int>(std::round(progress * kWidth)), 0, kWidth);
    const std::string bar = std::string(static_cast<size_t>(filled), '=') + std::string(static_cast<size_t>(kWidth - filled), ' ');
    std::ostringstream line;
    line << "\r[" << bar << "] " << percent << "% " << detail;

    const std::string text = line.str();
    std::fprintf(stderr, "%s", text.c_str());
    if (text.size() < last_line_width_) {
      std::fprintf(stderr, "%s", std::string(last_line_width_ - text.size(), ' ').c_str());
    }
    std::fflush(stderr);

    rendered_ = true;
    last_percent_ = percent;
    last_detail_ = detail;
    last_line_width_ = text.size();
  }

  bool enabled_ = false;
  bool rendered_ = false;
  int last_percent_ = -1;
  size_t last_line_width_ = 0;
  std::string last_detail_;
  std::mutex mutex_;
};

TerminalRouteProgress::TerminalRouteProgress(bool enabled)
  : impl_(std::make_unique<Impl>(enabled)) {}

TerminalRouteProgress::~TerminalRouteProgress() {
  finish();
}

void TerminalRouteProgress::update(const RouteLoadProgress &progress) {
  impl_->update(progress);
}

void TerminalRouteProgress::finish() {
  impl_->finish();
}

struct AsyncRouteLoader::Impl {
  explicit Impl(bool enable_terminal_progress)
      : terminal_progress(enable_terminal_progress) {}

  ~Impl() {
    join();
  }

  void start(const std::string &route_name_value, const std::string &data_dir_value, const std::string &dbc_name_value) {
    join();
    {
      std::lock_guard<std::mutex> lock(mutex);
      route_name = route_name_value;
      data_dir = data_dir_value;
      dbc_name = dbc_name_value;
      result.reset();
      error_text.clear();
    }
    active.store(!route_name_value.empty());
    completed.store(route_name_value.empty());
    success.store(route_name_value.empty());
    total_segments.store(0);
    segments_downloaded.store(0);
    segments_parsed.store(0);
    if (route_name_value.empty()) {
      return;
    }

    worker = std::thread([this]() {
      try {
        RouteData route_data = load_route_data(route_name, data_dir, dbc_name, [this](const RouteLoadProgress &progress) {
          total_segments.store(progress.total_segments > 0 ? progress.total_segments : progress.segment_count);
          segments_downloaded.store(progress.segments_downloaded);
          segments_parsed.store(progress.segments_parsed);
          terminal_progress.update(progress);
        });
        {
          std::lock_guard<std::mutex> lock(mutex);
          result = std::make_unique<RouteData>(std::move(route_data));
          error_text.clear();
        }
        success.store(true);
      } catch (const std::exception &err) {
        std::lock_guard<std::mutex> lock(mutex);
        result.reset();
        error_text = err.what();
        success.store(false);
      }
      active.store(false);
      completed.store(true);
      terminal_progress.finish();
    });
  }

  RouteLoadSnapshot snapshot() const {
    RouteLoadSnapshot snapshot;
    snapshot.active = active.load();
    snapshot.total_segments = total_segments.load();
    snapshot.segments_downloaded = segments_downloaded.load();
    snapshot.segments_parsed = segments_parsed.load();
    return snapshot;
  }

  bool consume(RouteData *route_data, std::string *error_text_out) {
    if (!completed.load()) return false;
    join();
    std::lock_guard<std::mutex> lock(mutex);
    completed.store(false);
    if (result) {
      *route_data = std::move(*result);
      result.reset();
      if (error_text_out != nullptr) {
        error_text_out->clear();
      }
      return true;
    }
    if (error_text_out != nullptr) {
      *error_text_out = error_text;
    }
    return true;
  }

  void join() {
    if (worker.joinable()) {
      worker.join();
    }
  }

  mutable std::mutex mutex;
  std::thread worker;
  std::unique_ptr<RouteData> result;
  std::string route_name;
  std::string data_dir;
  std::string dbc_name;
  std::string error_text;
  std::atomic<bool> active{false};
  std::atomic<bool> completed{false};
  std::atomic<bool> success{false};
  std::atomic<size_t> total_segments{0};
  std::atomic<size_t> segments_downloaded{0};
  std::atomic<size_t> segments_parsed{0};
  TerminalRouteProgress terminal_progress;
};

AsyncRouteLoader::AsyncRouteLoader(bool enable_terminal_progress)
  : impl_(std::make_unique<Impl>(enable_terminal_progress)) {}

AsyncRouteLoader::~AsyncRouteLoader() = default;

void AsyncRouteLoader::start(const std::string &route_name, const std::string &data_dir, const std::string &dbc_name) {
  impl_->start(route_name, data_dir, dbc_name);
}

RouteLoadSnapshot AsyncRouteLoader::snapshot() const {
  return impl_->snapshot();
}

bool AsyncRouteLoader::consume(RouteData *route_data, std::string *error_text) {
  return impl_->consume(route_data, error_text);
}

struct StreamPoller::Impl {
  ~Impl() {
    stop();
  }

  void start(const StreamSourceConfig &requested_source,
             double requested_buffer_seconds,
             const std::string &dbc_name,
             std::optional<double> time_offset) {
    stop();
    {
      std::lock_guard<std::mutex> lock(mutex);
      pending = {};
      pending_series_slots.clear();
      pending_can_slots.clear();
      error_text.clear();
      source = requested_source;
      if (source.kind == StreamSourceKind::CerealLocal) {
        source.address = "127.0.0.1";
      } else if (source.kind == StreamSourceKind::CerealRemote) {
        source.address = normalize_stream_address(source.address);
      }
      buffer_seconds = std::max(1.0, requested_buffer_seconds);
      latest_dbc_name = dbc_name;
      latest_car_fingerprint.clear();
    }
    received_messages.store(0);
    connected.store(false);
    paused.store(false);
    running.store(true);
    worker = std::thread([this, dbc_name, time_offset]() {
      try {
        StreamAccumulator accumulator(dbc_name, time_offset);
        switch (source.kind) {
          case StreamSourceKind::CerealLocal:
          case StreamSourceKind::CerealRemote:
            run_cereal_source(&accumulator);
            break;
        }
      } catch (const std::exception &err) {
        std::lock_guard<std::mutex> lock(mutex);
        error_text = err.what();
      }
      connected.store(false);
      running.store(false);
    });
  }

  void setPaused(bool paused_value) {
    paused.store(paused_value);
    if (paused_value) {
      std::lock_guard<std::mutex> lock(mutex);
      pending = {};
      pending_series_slots.clear();
      pending_can_slots.clear();
      error_text.clear();
    }
  }

  void set_error_text(std::string text) {
    std::lock_guard<std::mutex> lock(mutex);
    error_text = std::move(text);
  }

  void clear_error_text() {
    std::lock_guard<std::mutex> lock(mutex);
    error_text.clear();
  }

  void stop() {
    running.store(false);
    paused.store(false);
    if (worker.joinable()) {
      worker.join();
    }
    connected.store(false);
  }

  StreamPollSnapshot snapshot() const {
    StreamPollSnapshot out;
    out.active = running.load();
    out.connected = connected.load();
    out.paused = paused.load();
    out.source_kind = source.kind;
    out.source_label = stream_source_target_label(source);
    out.buffer_seconds = buffer_seconds;
    out.received_messages = received_messages.load();
    std::lock_guard<std::mutex> lock(mutex);
    out.dbc_name = latest_dbc_name;
    out.car_fingerprint = latest_car_fingerprint;
    return out;
  }

  bool consume(StreamExtractBatch *batch, std::string *out_error_text) {
    std::lock_guard<std::mutex> lock(mutex);
    const bool has_error = !error_text.empty();
    const bool has_batch = stream_batch_has_data(pending);
    if (!has_error && !has_batch) return false;
    if (batch != nullptr) {
      *batch = std::move(pending);
      pending = {};
      pending_series_slots.clear();
      pending_can_slots.clear();
    }
    if (out_error_text != nullptr) {
      *out_error_text = error_text;
      error_text.clear();
    }
    return true;
  }

  static void merge_route_series(RouteSeries *dst, RouteSeries *src) {
    if (src->times.empty()) {
      return;
    }
    if (dst->path.empty()) {
      dst->path = src->path;
    }
    dst->times.insert(dst->times.end(), src->times.begin(), src->times.end());
    dst->values.insert(dst->values.end(), src->values.begin(), src->values.end());
  }

  static void merge_can_message_data(CanMessageData *dst, CanMessageData *src) {
    if (src->samples.empty()) {
      return;
    }
    if (dst->samples.empty()) {
      *dst = std::move(*src);
      return;
    }
    dst->samples.insert(dst->samples.end(),
                        std::make_move_iterator(src->samples.begin()),
                        std::make_move_iterator(src->samples.end()));
  }

  static void merge_batch(StreamExtractBatch *dst,
                          std::unordered_map<std::string, size_t> *series_slots,
                          std::unordered_map<CanMessageId, size_t, CanMessageIdHash> *can_slots,
                          StreamExtractBatch *src) {
    for (RouteSeries &series : src->series) {
      auto [it, inserted] = series_slots->try_emplace(series.path, dst->series.size());
      if (inserted) {
        dst->series.push_back(RouteSeries{.path = series.path});
      }
      merge_route_series(&dst->series[it->second], &series);
    }
    for (CanMessageData &message : src->can_messages) {
      auto [it, inserted] = can_slots->try_emplace(message.id, dst->can_messages.size());
      if (inserted) {
        dst->can_messages.push_back(CanMessageData{.id = message.id});
      }
      merge_can_message_data(&dst->can_messages[it->second], &message);
    }
    if (!src->logs.empty()) {
      dst->logs.insert(dst->logs.end(),
                       std::make_move_iterator(src->logs.begin()),
                       std::make_move_iterator(src->logs.end()));
    }
    if (!src->timeline.empty()) {
      dst->timeline.insert(dst->timeline.end(),
                           std::make_move_iterator(src->timeline.begin()),
                           std::make_move_iterator(src->timeline.end()));
    }
    for (auto &[path, info] : src->enum_info) {
      dst->enum_info[path] = std::move(info);
    }
    if (!src->car_fingerprint.empty()) {
      dst->car_fingerprint = src->car_fingerprint;
    }
    if (!src->dbc_name.empty()) {
      dst->dbc_name = src->dbc_name;
    }
  }

  void publish_batch(StreamAccumulator *accumulator) {
    StreamExtractBatch batch = accumulator->takeBatch();
    if (!stream_batch_has_data(batch)) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex);
    merge_batch(&pending, &pending_series_slots, &pending_can_slots, &batch);
    latest_dbc_name = pending.dbc_name;
    latest_car_fingerprint = pending.car_fingerprint;
  }

  void run_cereal_source(StreamAccumulator *accumulator) {
    if (source.kind == StreamSourceKind::CerealRemote) {
      setenv("ZMQ", "1", 1);
    } else {
      unsetenv("ZMQ");
    }

    std::unique_ptr<Context> context(Context::create());
    std::unique_ptr<Poller> poller(Poller::create());
    std::vector<std::unique_ptr<SubSocket>> sockets;
    sockets.reserve(services.size());
    for (const auto &[name, info] : services) {
      if (!should_subscribe_stream_service(name)) continue;
      std::unique_ptr<SubSocket> socket(
        SubSocket::create(context.get(), name.c_str(), source.address.c_str(), false, true, info.queue_size));
      if (socket == nullptr) continue;
      socket->setTimeout(0);
      poller->registerSocket(socket.get());
      sockets.push_back(std::move(socket));
    }
    if (sockets.empty()) throw std::runtime_error("Failed to connect to any cereal service");
    connected.store(true);

    while (running.load()) {
      std::vector<SubSocket *> ready = poller->poll(1);
      for (SubSocket *socket : ready) {
        while (running.load()) {
          std::unique_ptr<Message> msg(socket->receive(true));
          if (!msg) break;
          const size_t size = msg->getSize();
          if (size < sizeof(capnp::word) || (size % sizeof(capnp::word)) != 0) {
            continue;
          }
          if (paused.load()) {
            received_messages.fetch_add(1);
            continue;
          }
          kj::ArrayPtr<const capnp::word> data(reinterpret_cast<const capnp::word *>(msg->getData()),
                                               size / sizeof(capnp::word));
          accumulator->appendEvent(data);
          received_messages.fetch_add(1);
        }
      }
      publish_batch(accumulator);
    }
  }

  mutable std::mutex mutex;
  std::thread worker;
  std::atomic<bool> running{false};
  std::atomic<bool> connected{false};
  std::atomic<bool> paused{false};
  std::atomic<uint64_t> received_messages{0};
  StreamExtractBatch pending;
  std::unordered_map<std::string, size_t> pending_series_slots;
  std::unordered_map<CanMessageId, size_t, CanMessageIdHash> pending_can_slots;
  std::string error_text;
  StreamSourceConfig source;
  std::string latest_dbc_name;
  std::string latest_car_fingerprint;
  double buffer_seconds = 30.0;
};

StreamPoller::StreamPoller()
  : impl_(std::make_unique<Impl>()) {}

StreamPoller::~StreamPoller() = default;

void StreamPoller::start(const StreamSourceConfig &source,
                         double buffer_seconds,
                         const std::string &dbc_name,
                         std::optional<double> time_offset) {
  impl_->start(source, buffer_seconds, dbc_name, time_offset);
}

void StreamPoller::setPaused(bool paused) {
  impl_->setPaused(paused);
}

void StreamPoller::stop() {
  impl_->stop();
}

StreamPollSnapshot StreamPoller::snapshot() const {
  return impl_->snapshot();
}

bool StreamPoller::consume(StreamExtractBatch *batch, std::string *error_text) {
  return impl_->consume(batch, error_text);
}

struct CameraFeedView::Impl {
  struct RequestKey {
    int segment = -1;
    int decode_index = -1;
  };

  struct DecodeRequest {
    RequestKey key;
    std::string path;
    uint64_t serial = 0;
    uint64_t generation = 0;
  };

  struct PreloadTask {
    int segment = -1;
    std::string path;
    uint64_t generation = 0;
  };

  struct DecodeResult {
    RequestKey key;
    bool success = false;
    int width = 0;
    int height = 0;
    double decode_ms = 0.0;
    std::vector<uint8_t> rgba;
  };

  static constexpr float kDefaultAspect = 1208.0f / 1928.0f;
  static constexpr size_t kCachedFrames = 8;
  static constexpr int kPrefetchAhead = 2;
  static constexpr int kImmediateNearbyFrameDistance = 8;
  static constexpr int kPreloadWorkerCount = 2;

  Impl() {
    demand_worker = std::thread([this]() { demand_worker_loop(); });
    for (int i = 0; i < kPreloadWorkerCount; ++i) {
      preload_workers.emplace_back([this]() { preload_worker_loop(); });
    }
  }

  ~Impl() {
    stop.store(true);
    cv.notify_all();
    if (demand_worker.joinable()) {
      demand_worker.join();
    }
    for (std::thread &worker : preload_workers) {
      if (worker.joinable()) {
        worker.join();
      }
    }
    destroy_texture();
  }

  void setRouteData(const RouteData &route_data) {
    setCameraIndex(route_data.road_camera, CameraViewKind::Road);
  }

  void setCameraIndex(const CameraFeedIndex &camera_index, CameraViewKind view) {
    destroy_texture();
    {
      std::lock_guard<std::mutex> lock(mutex);
      route_index = camera_index;
      camera_view = view;
      pending_request.reset();
      pending_result.reset();
      cached_results.clear();
      preload_queue.clear();
      preload_focus_segment = -1;
      ++route_generation;
      latest_request_serial = 0;
      int initial_focus_segment = -1;
      if (!route_index.entries.empty()) {
        initial_focus_segment = route_index.entries.front().segment;
      } else {
        for (const CameraSegmentFile &segment_file : route_index.segment_files) {
          if (!segment_file.path.empty()) {
            initial_focus_segment = segment_file.segment;
            break;
          }
        }
      }
      if (initial_focus_segment >= 0) {
        rebuild_preload_queue_locked(initial_focus_segment);
      }
    }
    abort_demand_work.store(true);
    abort_preload_work.store(true);
    active_request.reset();
    displayed_request.reset();
    failed_request.reset();
    frame_width = 0;
    frame_height = 0;
    cv.notify_all();
  }

  void update(double tracker_time) {
    upload_pending_result();
    std::optional<DecodeRequest> request = request_for_time(tracker_time);
    if (!request.has_value()) {
      return;
    }
    if (same_request(active_request, request->key) || same_request(displayed_request, request->key) ||
        same_request(failed_request, request->key)) {
      return;
    }
    if (try_upload_cached_result(request->key)) {
      return;
    }
    try_upload_nearby_cached_result(request->key);

    bool focus_changed = false;
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (preload_focus_segment != request->key.segment) {
        rebuild_preload_queue_locked(request->key.segment);
        focus_changed = true;
      }
      request->serial = ++latest_request_serial;
      request->generation = route_generation;
      pending_request = request;
    }
    abort_demand_work.store(true);
    if (focus_changed) {
      abort_preload_work.store(true);
    }
    active_request = request->key;
    failed_request.reset();
    cv.notify_all();
  }

  void draw(float width, bool loading) {
    const float preview_width = std::max(1.0f, width);
    const float preview_height = preview_width * preview_aspect();
    drawSized(ImVec2(preview_width, preview_height), loading, false);
    ImGui::Spacing();
  }

  void drawSized(ImVec2 size, bool loading, bool fit_to_pane) {
    size.x = std::max(1.0f, size.x);
    size.y = std::max(1.0f, size.y);
    const float aspect = preview_aspect();
    ImVec2 frame_size = size;
    ImVec2 top_left = ImGui::GetCursorScreenPos();
    ImVec2 uv0(0.0f, 0.0f);
    ImVec2 uv1(1.0f, 1.0f);
    if (aspect > 0.0f && !fit_to_pane) {
      frame_size.y = std::min(size.y, size.x * aspect);
      frame_size.x = std::min(size.x, frame_size.y / aspect);
      top_left = ImVec2(top_left.x + (size.x - frame_size.x) * 0.5f,
                        top_left.y + (size.y - frame_size.y) * 0.5f);
    } else if (aspect > 0.0f && fit_to_pane) {
      const float src_aspect = 1.0f / aspect;
      const float dst_aspect = size.x / size.y;
      if (dst_aspect > src_aspect) {
        const float visible_v = std::clamp(src_aspect / dst_aspect, 0.0f, 1.0f);
        const float v_pad = (1.0f - visible_v) * 0.5f;
        uv0.y = v_pad;
        uv1.y = 1.0f - v_pad;
      } else if (dst_aspect < src_aspect) {
        const float visible_u = std::clamp(dst_aspect / src_aspect, 0.0f, 1.0f);
        const float u_pad = (1.0f - visible_u) * 0.5f;
        uv0.x = u_pad;
        uv1.x = 1.0f - u_pad;
      }
    }
    ImGui::InvisibleButton("##camera_feed_sized", size);
    if (texture != 0) {
      ImGui::GetWindowDrawList()->AddImage(static_cast<ImTextureID>(texture),
                                           top_left,
                                           ImVec2(top_left.x + frame_size.x, top_left.y + frame_size.y),
                                           uv0,
                                           uv1);
    } else {
      ImDrawList *draw_list = ImGui::GetWindowDrawList();
      draw_list->AddRectFilled(top_left, ImVec2(top_left.x + frame_size.x, top_left.y + frame_size.y), IM_COL32(45, 47, 50, 255));
      draw_list->AddRect(top_left, ImVec2(top_left.x + frame_size.x, top_left.y + frame_size.y), IM_COL32(95, 100, 106, 255));

      const char *label = (loading || has_video_source()) ? "loading" : "no video";
      const ImVec2 text_size = ImGui::CalcTextSize(label);
      const ImVec2 text_pos(top_left.x + (frame_size.x - text_size.x) * 0.5f,
                            top_left.y + (frame_size.y - text_size.y) * 0.5f);
      draw_list->AddText(text_pos, IM_COL32(187, 187, 187, 255), label);
    }
  }

  static bool same_request(const std::optional<RequestKey> &lhs, const RequestKey &rhs) {
    return lhs.has_value() && lhs->segment == rhs.segment && lhs->decode_index == rhs.decode_index;
  }

  bool has_video_source() const {
    std::lock_guard<std::mutex> lock(mutex);
    return !route_index.entries.empty() && !route_index.segment_files.empty();
  }

  float preview_aspect() const {
    if (frame_width > 0 && frame_height > 0) return static_cast<float>(frame_height) / static_cast<float>(frame_width);
    return kDefaultAspect;
  }

  std::optional<DecodeRequest> request_for_time(double tracker_time) const {
    std::lock_guard<std::mutex> lock(mutex);
    if (route_index.entries.empty()) return std::nullopt;

    auto it = std::lower_bound(route_index.entries.begin(), route_index.entries.end(), tracker_time,
                               [](const CameraFrameIndexEntry &entry, double tm) {
                                 return entry.timestamp < tm;
                               });
    if (it == route_index.entries.end()) {
      it = std::prev(route_index.entries.end());
    } else if (it != route_index.entries.begin()) {
      const auto prev = std::prev(it);
      if (std::abs(prev->timestamp - tracker_time) <= std::abs(it->timestamp - tracker_time)) {
        it = prev;
      }
    }

    auto path_it = std::find_if(route_index.segment_files.begin(), route_index.segment_files.end(),
                                [&](const CameraSegmentFile &segment) {
                                  return segment.segment == it->segment && !segment.path.empty();
                                });
    if (path_it == route_index.segment_files.end()) return std::nullopt;

    return DecodeRequest{
      .key = RequestKey{.segment = it->segment, .decode_index = it->decode_index},
      .path = path_it->path,
    };
  }

  void upload_pending_result() {
    std::optional<DecodeResult> result;
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (!pending_result.has_value()) {
        return;
      }
      result = std::move(pending_result);
      pending_result.reset();
    }

    active_request.reset();
    if (!result->success || result->rgba.empty() || result->width <= 0 || result->height <= 0) {
      failed_request = result->key;
      return;
    }

    upload_result(*result);
  }

  void upload_result(const DecodeResult &result) {
    remember_cached_result(result);

    const bool new_size = texture_width != result.width || texture_height != result.height;
    if (texture == 0) {
      glGenTextures(1, &texture);
    }
    glBindTexture(GL_TEXTURE_2D, texture);
    if (new_size) {
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
      glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, result.width, result.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, result.rgba.data());
      texture_width = result.width;
      texture_height = result.height;
    } else {
      glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, result.width, result.height, GL_RGBA, GL_UNSIGNED_BYTE, result.rgba.data());
    }
    glBindTexture(GL_TEXTURE_2D, 0);

    frame_width = result.width;
    frame_height = result.height;
    displayed_request = result.key;
    failed_request.reset();
  }

  bool try_upload_cached_result(const RequestKey &key) {
    std::optional<DecodeResult> result;
    {
      std::lock_guard<std::mutex> lock(mutex);
      auto it = std::find_if(cached_results.begin(), cached_results.end(), [&](const DecodeResult &cached) {
        return cached.key.segment == key.segment && cached.key.decode_index == key.decode_index;
      });
      if (it == cached_results.end()) {
        return false;
      }
      result = *it;
    }
    active_request.reset();
    upload_result(*result);
    return true;
  }

  bool try_upload_nearby_cached_result(const RequestKey &key) {
    std::optional<DecodeResult> result;
    int best_distance = std::numeric_limits<int>::max();
    {
      std::lock_guard<std::mutex> lock(mutex);
      for (const DecodeResult &cached : cached_results) {
        if (cached.key.segment != key.segment) continue;
        const int distance = std::abs(cached.key.decode_index - key.decode_index);
        if (distance == 0 || distance > kImmediateNearbyFrameDistance || distance >= best_distance) continue;
        best_distance = distance;
        result = cached;
      }
    }
    if (!result.has_value()) {
      return false;
    }
    upload_result(*result);
    return true;
  }

  void remember_cached_result(const DecodeResult &result) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = std::find_if(cached_results.begin(), cached_results.end(), [&](const DecodeResult &cached) {
      return cached.key.segment == result.key.segment && cached.key.decode_index == result.key.decode_index;
    });
    if (it != cached_results.end()) {
      cached_results.erase(it);
    }
    cached_results.push_front(result);
    while (cached_results.size() > kCachedFrames) {
      cached_results.pop_back();
    }
  }

  void destroy_texture() {
    if (texture != 0 && g_glfw_alive.load() && glfwGetCurrentContext() != nullptr) {
      glDeleteTextures(1, &texture);
    }
    texture = 0;
    texture_width = 0;
    texture_height = 0;
    frame_width = 0;
    frame_height = 0;
  }

  static bool ensure_decode_buffer(FrameReader *reader, VisionBuf *buf, bool &allocated, int &w, int &h) {
    if (!reader) return false;
    if (allocated && w == reader->width && h == reader->height) return true;
    if (allocated) { buf->free(); allocated = false; }
    const auto [stride, y_height, _uv_height, size] = get_nv12_info(reader->width, reader->height);
    buf->allocate(size);
    buf->init_yuv(reader->width, reader->height, stride, stride * y_height);
    w = reader->width;
    h = reader->height;
    allocated = true;
    return true;
  }

  void publish_result(const DecodeRequest &request, DecodeResult result) {
    std::lock_guard<std::mutex> lock(mutex);
    if (!pending_request.has_value() || pending_request->serial != request.serial ||
        pending_request->generation != request.generation) {
      return;
    }
    pending_result = std::move(result);
  }

  bool has_newer_pending_request(uint64_t serial) const {
    std::lock_guard<std::mutex> lock(mutex);
    return pending_request.has_value() && pending_request->serial != serial;
  }

  void rebuild_preload_queue_locked(int focus_segment) {
    std::vector<PreloadTask> ordered;
    ordered.reserve(route_index.segment_files.size());
    for (const CameraSegmentFile &segment_file : route_index.segment_files) {
      if (segment_file.path.empty()) continue;
      if (segment_file.segment == focus_segment) continue;
      ordered.push_back(PreloadTask{
        .segment = segment_file.segment,
        .path = segment_file.path,
        .generation = route_generation,
      });
    }
    std::sort(ordered.begin(), ordered.end(), [&](const PreloadTask &a, const PreloadTask &b) {
      const int distance_a = std::abs(a.segment - focus_segment);
      const int distance_b = std::abs(b.segment - focus_segment);
      if (distance_a != distance_b) return distance_a < distance_b;
      return a.segment < b.segment;
    });
    preload_queue.assign(ordered.begin(), ordered.end());
    preload_focus_segment = focus_segment;
  }

  std::shared_ptr<FrameReader> find_loaded_reader_locked(int segment, uint64_t generation) {
    if (readers_generation != generation) {
      readers.clear();
      loading_segments.clear();
      readers_generation = generation;
    }
    auto it = readers.find(segment);
    return it != readers.end() ? it->second : nullptr;
  }

  std::shared_ptr<FrameReader> ensure_reader_loaded(int segment,
                                                    const std::string &path,
                                                    uint64_t generation,
                                                    const char *reason,
                                                    std::atomic<bool> *abort_flag,
                                                    bool wait_for_inflight) {
    while (!stop.load()) {
      {
        std::unique_lock<std::mutex> lock(readers_mutex);
        if (std::shared_ptr<FrameReader> cached = find_loaded_reader_locked(segment, generation)) {
          return cached;
        }
        if (loading_segments.find(segment) != loading_segments.end()) {
          if (!wait_for_inflight) {
            return nullptr;
          }
          readers_cv.wait(lock, [&]() {
            return stop.load()
                || readers_generation != generation
                || loading_segments.find(segment) == loading_segments.end();
          });
          continue;
        }
        loading_segments.insert(segment);
      }

      auto reader = std::make_shared<FrameReader>();
      const auto load_begin = std::chrono::steady_clock::now();
      const bool loaded = reader->load(decoder_camera_type(camera_view), path, false, abort_flag, true);

      {
        std::lock_guard<std::mutex> lock(readers_mutex);
        if (readers_generation != generation) {
          readers.clear();
          loading_segments.clear();
          readers_generation = generation;
        }
        loading_segments.erase(segment);
        if (loaded) {
          readers[segment] = reader;
        }
      }
      readers_cv.notify_all();

      if (!loaded) {
        return nullptr;
      }
      if (kLogCameraTimings) {
        const double load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - load_begin).count();
        std::fprintf(stderr, "camera[%s] %s-load seg=%d %.1fms\n",
                     camera_view_spec(camera_view).runtime_name, reason, segment, load_ms);
      }
      return reader;
    }
    return nullptr;
  }

  void preload_worker_loop() {
    while (true) {
      std::optional<PreloadTask> preload;
      {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&]() { return stop.load() || !preload_queue.empty(); });
        if (stop.load()) {
          break;
        }
        preload = preload_queue.front();
        preload_queue.pop_front();
      }

      abort_preload_work.store(false);
      {
        std::lock_guard<std::mutex> lock(readers_mutex);
        if (find_loaded_reader_locked(preload->segment, preload->generation)) {
          continue;
        }
      }
      ensure_reader_loaded(preload->segment, preload->path, preload->generation, "preload",
                           &abort_preload_work, false);
    }
  }

  void demand_worker_loop() {
    uint64_t handled_serial = 0;
    VisionBuf decode_buffer;
    bool buffer_allocated = false;
    int buffer_width = 0;
    int buffer_height = 0;

    while (true) {
      std::optional<DecodeRequest> request;
      {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&]() {
          return stop.load() || (pending_request.has_value() && pending_request->serial != handled_serial);
        });
        if (stop.load()) break;
        request = *pending_request;
        handled_serial = request->serial;
      }

      abort_demand_work.store(false);

      DecodeResult result{.key = request->key};
      std::shared_ptr<FrameReader> reader = ensure_reader_loaded(request->key.segment, request->path,
                                                                 request->generation, "demand",
                                                                 &abort_demand_work, true);
      if (!reader) {
        publish_result(*request, std::move(result));
        continue;
      }
      if (has_newer_pending_request(request->serial)) {
        continue;
      }

      const auto decode_begin = std::chrono::steady_clock::now();
      if (!ensure_decode_buffer(reader.get(), &decode_buffer, buffer_allocated, buffer_width, buffer_height) ||
          !reader->get(request->key.decode_index, &decode_buffer)) {
        publish_result(*request, std::move(result));
        continue;
      }

      result.width = reader->width;
      result.height = reader->height;
      result.rgba.resize(static_cast<size_t>(result.width) * static_cast<size_t>(result.height) * 4U, 0);
      libyuv::NV12ToABGR(decode_buffer.y,
                         static_cast<int>(decode_buffer.stride),
                         decode_buffer.uv,
                         static_cast<int>(decode_buffer.stride),
                         result.rgba.data(),
                         result.width * 4,
                         result.width,
                         result.height);
      result.success = true;
      result.decode_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - decode_begin).count();
      publish_result(*request, std::move(result));

      for (int offset = 1; offset <= kPrefetchAhead; ++offset) {
        if (stop.load() || has_newer_pending_request(request->serial)) {
          break;
        }
        const int next_index = request->key.decode_index + offset;
        if (next_index < 0 || next_index >= static_cast<int>(reader->getFrameCount())) {
          break;
        }
        if (!reader->get(next_index, &decode_buffer)) {
          break;
        }
        DecodeResult prefetched{
          .key = RequestKey{.segment = request->key.segment, .decode_index = next_index},
          .success = true,
          .width = reader->width,
          .height = reader->height,
        };
        prefetched.rgba.resize(static_cast<size_t>(prefetched.width) * static_cast<size_t>(prefetched.height) * 4U, 0);
        libyuv::NV12ToABGR(decode_buffer.y,
                           static_cast<int>(decode_buffer.stride),
                           decode_buffer.uv,
                           static_cast<int>(decode_buffer.stride),
                           prefetched.rgba.data(),
                           prefetched.width * 4,
                           prefetched.width,
                           prefetched.height);
        remember_cached_result(prefetched);
      }
    }

    if (buffer_allocated) {
      decode_buffer.free();
    }
  }

  mutable std::mutex mutex;
  std::condition_variable cv;
  std::thread demand_worker;
  std::vector<std::thread> preload_workers;
  std::atomic<bool> stop{false};
  std::atomic<bool> abort_demand_work{false};
  std::atomic<bool> abort_preload_work{false};
  CameraFeedIndex route_index;
  CameraViewKind camera_view = CameraViewKind::Road;
  std::optional<DecodeRequest> pending_request;
  std::optional<DecodeResult> pending_result;
  std::deque<PreloadTask> preload_queue;
  int preload_focus_segment = -1;
  std::deque<DecodeResult> cached_results;
  uint64_t latest_request_serial = 0;
  uint64_t route_generation = 1;
  std::optional<RequestKey> active_request;
  std::optional<RequestKey> displayed_request;
  std::optional<RequestKey> failed_request;
  std::mutex readers_mutex;
  std::condition_variable readers_cv;
  std::unordered_map<int, std::shared_ptr<FrameReader>> readers;
  std::unordered_set<int> loading_segments;
  uint64_t readers_generation = 0;
  GLuint texture = 0;
  int texture_width = 0;
  int texture_height = 0;
  int frame_width = 0;
  int frame_height = 0;
};

CameraFeedView::CameraFeedView()
  : impl_(std::make_unique<Impl>()) {}

CameraFeedView::~CameraFeedView() = default;

void CameraFeedView::setRouteData(const RouteData &route_data) {
  impl_->setRouteData(route_data);
}

void CameraFeedView::setCameraIndex(const CameraFeedIndex &camera_index, CameraViewKind view) {
  impl_->setCameraIndex(camera_index, view);
}

void CameraFeedView::update(double tracker_time) {
  impl_->update(tracker_time);
}

void CameraFeedView::draw(float width, bool loading) {
  impl_->draw(width, loading);
}

void CameraFeedView::drawSized(ImVec2 size, bool loading, bool fit_to_pane) {
  impl_->drawSized(size, loading, fit_to_pane);
}
