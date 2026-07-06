#include "tools/loggy/backend/session.h"

#include "tools/loggy/backend/dbc/dbcmanager.h"
#include "tools/loggy/car_fingerprint_to_dbc.h"

#include "json11/json11.hpp"

#include <filesystem>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <iterator>
#include <optional>
#include <sstream>
#include <stdexcept>

namespace loggy {
namespace {

namespace fs = std::filesystem;

fs::path resolve_layout_path(const std::string &layout) {
  if (layout.empty()) return {};

  const fs::path direct(layout);
  if (fs::exists(direct)) return fs::absolute(direct);

  fs::path candidate = layouts_dir() / layout;
  if (candidate.extension().empty()) candidate.replace_extension(".json");
  if (fs::exists(candidate)) return candidate;

  throw std::runtime_error("Unknown layout: " + layout);
}

fs::path resolve_preset_layout_path(const std::string &preset) {
  if (preset.empty()) return {};
  fs::path candidate = layouts_dir() / preset;
  if (candidate.extension().empty()) candidate.replace_extension(".json");
  return fs::exists(candidate) ? candidate : fs::path{};
}

bool timeline_span_less(const TimelineSpan &a, const TimelineSpan &b) {
  if (std::abs(a.start_time - b.start_time) > 1.0e-9) return a.start_time < b.start_time;
  if (std::abs(a.end_time - b.end_time) > 1.0e-9) return a.end_time < b.end_time;
  return static_cast<int>(a.kind) < static_cast<int>(b.kind);
}

std::vector<TimelineSpan> merge_timeline_spans(std::vector<TimelineSpan> spans) {
  spans.erase(std::remove_if(spans.begin(), spans.end(), [](const TimelineSpan &span) {
    return span.kind == TimelineSpanKind::None || !std::isfinite(span.start_time) || !std::isfinite(span.end_time);
  }), spans.end());
  for (TimelineSpan &span : spans) {
    if (span.end_time < span.start_time) std::swap(span.start_time, span.end_time);
  }
  std::sort(spans.begin(), spans.end(), timeline_span_less);

  std::vector<TimelineSpan> merged;
  merged.reserve(spans.size());
  for (const TimelineSpan &span : spans) {
    if (!merged.empty() && merged.back().kind == span.kind && span.start_time <= merged.back().end_time + 1.0e-9) {
      merged.back().end_time = std::max(merged.back().end_time, span.end_time);
      continue;
    }
    merged.push_back(span);
  }
  return merged;
}

void append_and_sort_logs(std::vector<LogEntry> *dst, std::vector<LogEntry> incoming) {
  if (dst == nullptr || incoming.empty()) return;
  dst->insert(dst->end(),
              std::make_move_iterator(incoming.begin()),
              std::make_move_iterator(incoming.end()));
  std::stable_sort(dst->begin(), dst->end(), [](const LogEntry &a, const LogEntry &b) {
    if (std::abs(a.mono_time - b.mono_time) > 1.0e-9) return a.mono_time < b.mono_time;
    return a.message < b.message;
  });
}

void trim_logs_before(std::vector<LogEntry> *logs, double cutoff_time) {
  if (logs == nullptr || !std::isfinite(cutoff_time)) return;
  logs->erase(std::remove_if(logs->begin(), logs->end(), [&](const LogEntry &entry) {
    return entry.mono_time < cutoff_time;
  }), logs->end());
}

void trim_timeline_spans_before(std::vector<TimelineSpan> *spans, double cutoff_time) {
  if (spans == nullptr || !std::isfinite(cutoff_time)) return;
  spans->erase(std::remove_if(spans->begin(), spans->end(), [&](const TimelineSpan &span) {
    return span.end_time < cutoff_time;
  }), spans->end());
  for (TimelineSpan &span : *spans) {
    if (span.start_time < cutoff_time) span.start_time = cutoff_time;
  }
}

std::string number_token(double value) {
  std::ostringstream out;
  out.precision(17);
  out << value;
  return out.str();
}

std::string trim_copy(std::string value) {
  auto is_space = [](unsigned char c) { return std::isspace(c) != 0; };
  value.erase(value.begin(), std::find_if_not(value.begin(), value.end(), is_space));
  value.erase(std::find_if_not(value.rbegin(), value.rend(), is_space).base(), value.end());
  return value;
}

std::string computed_operation_token(const json11::Json &series) {
  const std::string transform = series["transform"].string_value();
  std::string out = transform;
  if (series["derivative_dt"].is_number()) out += "|dt=" + number_token(series["derivative_dt"].number_value());
  if (series["scale"].is_number()) out += "|scale=" + number_token(series["scale"].number_value());
  if (series["offset"].is_number()) out += "|offset=" + number_token(series["offset"].number_value());
  return out;
}

std::string custom_python_operation_token(const json11::Json &custom_python) {
  std::string out = "python";
  out += "|linked=" + custom_python["linked_source"].string_value();
  out += "|globals=" + custom_python["globals_code"].string_value();
  out += "|function=" + custom_python["function_code"].string_value();
  for (const json11::Json &source : custom_python["additional_sources"].array_items()) {
    out += "|source=" + source.string_value();
  }
  return out;
}

bool append_unique_computed_spec(std::vector<ComputedSeriesSpec> *specs, ComputedSeriesSpec spec) {
  if (specs == nullptr || spec.output_path.empty()) return false;
  const auto existing = std::find_if(specs->begin(), specs->end(), [&](const ComputedSeriesSpec &item) {
    return item.output_path == spec.output_path;
  });
  if (existing != specs->end()) return false;
  specs->push_back(std::move(spec));
  return true;
}

bool normalize_plot_computed_series(PaneInstance *pane, std::vector<ComputedSeriesSpec> *specs) {
  if (pane == nullptr || pane->type != "plot") return false;
  std::string err;
  const json11::Json state = json11::Json::parse(pane->state_json, err);
  if (!err.empty() || !state.is_object() || !state["series"].is_array()) return false;

  bool changed = false;
  json11::Json::array normalized_series;
  normalized_series.reserve(state["series"].array_items().size());
  for (const json11::Json &item : state["series"].array_items()) {
    if (!item.is_object()) {
      normalized_series.push_back(item);
      continue;
    }

    json11::Json::object out = item.object_items();
    const std::string transform = item["transform"].string_value();
    const std::string source_path = item["path"].string_value();
    const json11::Json &custom_python = item["custom_python"];
    if (custom_python.is_object()) {
      const std::string linked_source = custom_python["linked_source"].is_string()
                                      ? custom_python["linked_source"].string_value()
                                      : source_path;
      const std::string label = item["label"].is_string() ? item["label"].string_value() :
                                (item["title"].is_string() ? item["title"].string_value() : std::string());
      const std::string operation = custom_python_operation_token(custom_python);
      ComputedSeriesSpec spec;
      spec.label = label;
      spec.kind = ComputedSeriesKind::CustomPython;
      spec.python.linked_source = linked_source;
      spec.python.globals_code = custom_python["globals_code"].string_value();
      spec.python.function_code = custom_python["function_code"].is_string()
                                ? custom_python["function_code"].string_value()
                                : "return value";
      for (const json11::Json &source : custom_python["additional_sources"].array_items()) {
        if (source.is_string()) spec.python.additional_sources.push_back(source.string_value());
      }
      spec.output_path = computed_output_path(linked_source.empty() ? source_path : linked_source, label, operation);
      append_unique_computed_spec(specs, spec);

      out["path"] = spec.output_path;
      out["computed_source_path"] = linked_source;
      out["computed_operation"] = operation;
      changed = true;
      normalized_series.push_back(json11::Json(out));
      continue;
    }

    const bool derivative = transform == "derivative";
    const bool scale = transform == "scale";
    if (!source_path.empty() && (derivative || scale)) {
      const std::string label = item["label"].is_string() ? item["label"].string_value() :
                                (item["title"].is_string() ? item["title"].string_value() : std::string());
      const std::string operation = computed_operation_token(item);
      ComputedSeriesSpec spec;
      spec.source_path = source_path;
      spec.label = label;
      spec.output_path = computed_output_path(source_path, label, operation);
      spec.kind = ComputedSeriesKind::Transform;
      spec.transform = derivative ? ComputedTransformKind::Derivative : ComputedTransformKind::Scale;
      spec.derivative_dt = item["derivative_dt"].is_number() ? item["derivative_dt"].number_value() : 0.0;
      spec.scale = item["scale"].is_number() ? item["scale"].number_value() : 1.0;
      spec.offset = item["offset"].is_number() ? item["offset"].number_value() : 0.0;
      append_unique_computed_spec(specs, spec);

      out["path"] = spec.output_path;
      out["computed_source_path"] = source_path;
      out["computed_operation"] = operation;
      out.erase("transform");
      out.erase("derivative_dt");
      out.erase("scale");
      out.erase("offset");
      changed = true;
    }
    normalized_series.push_back(json11::Json(out));
  }

  if (!changed) return false;
  json11::Json::object normalized_state = state.object_items();
  normalized_state["series"] = normalized_series;
  pane->state_json = json11::Json(normalized_state).dump();
  return true;
}

std::vector<ComputedSeriesSpec> normalize_workspace_computed_series(Workspace *workspace) {
  std::vector<ComputedSeriesSpec> specs;
  if (workspace == nullptr) return specs;
  for (WorkspaceTab &tab : workspace->tabs) {
    for (PaneInstance &pane : tab.panes) {
      normalize_plot_computed_series(&pane, &specs);
    }
  }
  return specs;
}

bool computed_dependencies_touched(const std::vector<ComputedSeriesSpec> &specs,
                                   const std::vector<std::string> &touched_paths) {
  if (specs.empty() || touched_paths.empty()) return false;
  for (const ComputedSeriesSpec &spec : specs) {
    for (const std::string &dependency : computed_dependencies(spec)) {
      if (std::binary_search(touched_paths.begin(), touched_paths.end(), dependency)) return true;
    }
  }
  return false;
}

bool camera_dependencies_touched(const std::vector<std::string> &touched_paths) {
  for (const std::string &path : touched_paths) {
    for (const CameraViewSpec &spec : camera_view_specs()) {
      const std::string prefix = "/" + std::string(spec.encode_index) + "/";
      if (path.rfind(prefix, 0) == 0) return true;
    }
  }
  return false;
}

void refresh_camera_indexes(const std::vector<RouteSegment> &segments,
                            const Store &store,
                            TimeRange range,
                            std::array<CameraFeedIndex, 4> *indexes) {
  if (indexes == nullptr) return;
  for (const CameraViewSpec &spec : camera_view_specs()) {
    (*indexes)[camera_view_index(spec.view)] = build_camera_feed_index(segments, store, spec.view, range);
  }
}

void extend_range(TimeRange *range, TimeRange incoming) {
  if (range == nullptr || !incoming.valid()) return;
  if (!range->valid()) {
    *range = incoming;
  } else {
    range->start_ = std::min(range->start_, incoming.start_);
    range->end = std::max(range->end, incoming.end);
  }
}

TimeRange live_display_range(TimeRange live_range, double buffer_seconds) {
  if (!live_range.valid()) return {};
  const double latest = live_range.end;
  const double start_ = std::max(0.0, latest - std::max(1.0, buffer_seconds));
  return {start_, std::max(start_ + 1.0, latest)};
}

LiveSourceConfig live_source_config_from_session(const SessionConfig &config) {
  LiveSourceConfig source;
  source.kind = config.stream_source_kind;
  source.panda_buses = config.stream_panda_buses;
  if (source.kind == LiveSourceKind::SocketCan || source.kind == LiveSourceKind::PandaUsb) {
    source.address = trim_copy(config.stream_address);
  } else if (source.kind == LiveSourceKind::CerealLocal) {
    source.address = "127.0.0.1";
  } else if (source.kind == LiveSourceKind::DeviceBridge) {
    source.address = normalize_live_stream_address(config.stream_address);
  } else {
    source.address = normalize_live_stream_address(config.stream_address);
  }
  source.buffer_seconds = std::max(1.0, config.stream_buffer_seconds);
  return source;
}

fs::path repo_root_path() {
#ifdef LOGGY_REPO_ROOT
  return fs::path(LOGGY_REPO_ROOT);
#else
  return fs::current_path();
#endif
}

fs::path default_opendbc_dbc_root() {
  return repo_root_path() / "opendbc_repo" / "opendbc" / "dbc";
}

fs::path generated_dbc_root() {
  return repo_root_path() / "openpilot" / "tools" / "loggy" / "generated_dbcs";
}

std::string dbc_name_for_fingerprint(std::string_view fingerprint) {
  return std::string(dbc_for_car_fingerprint(fingerprint));
}

std::optional<fs::path> resolve_dbc_reference(std::string_view dbc_reference, const LoggySettings &settings) {
  const std::string reference = trim_copy(std::string(dbc_reference));
  if (reference.empty()) return std::nullopt;

  const fs::path direct(reference);
  std::error_code ec;
  if (fs::exists(direct, ec)) return fs::absolute(direct, ec);

  std::string name = reference;
  if (direct.extension() == ".dbc") name = direct.stem().string();
  const std::array<fs::path, 3> roots = {{
    settings.opendbc_root.empty() ? default_opendbc_dbc_root() : fs::path(settings.opendbc_root),
    default_opendbc_dbc_root(),
    generated_dbc_root(),
  }};
  for (const fs::path &root : roots) {
    const fs::path candidate = root / (name + ".dbc");
    if (fs::exists(candidate, ec)) return fs::absolute(candidate, ec);
  }
  return std::nullopt;
}

void merge_drain_result(DrainResult *dst, DrainResult src) {
  if (dst == nullptr) return;
  dst->batches += src.batches;
  dst->series_chunks += src.series_chunks;
  dst->series_points += src.series_points;
  dst->can_chunks += src.can_chunks;
  dst->can_events += src.can_events;
  dst->touched_series_paths.insert(dst->touched_series_paths.end(),
                                   std::make_move_iterator(src.touched_series_paths.begin()),
                                   std::make_move_iterator(src.touched_series_paths.end()));
  std::sort(dst->touched_series_paths.begin(), dst->touched_series_paths.end());
  dst->touched_series_paths.erase(std::unique(dst->touched_series_paths.begin(), dst->touched_series_paths.end()),
                                  dst->touched_series_paths.end());
}

void merge_trim_result(DrainResult *dst, const StoreTrimResult &src) {
  if (dst == nullptr) return;
  dst->series_chunks += src.series_chunks_removed;
  dst->series_points += src.series_points_removed;
  dst->can_events += src.can_events_removed;
  dst->touched_series_paths.insert(dst->touched_series_paths.end(),
                                   src.touched_series_paths.begin(),
                                   src.touched_series_paths.end());
  std::sort(dst->touched_series_paths.begin(), dst->touched_series_paths.end());
  dst->touched_series_paths.erase(std::unique(dst->touched_series_paths.begin(), dst->touched_series_paths.end()),
                                  dst->touched_series_paths.end());
}

}  // namespace

Session::Session(SessionConfig cfg) : config(std::move(cfg)), scheduler(&store), route_ingest_(&scheduler) {
  config.stream_buffer_seconds = std::max(1.0, config.stream_buffer_seconds);
  config.stream_address = trim_copy(std::move(config.stream_address));
  for (PandaBusConfig &bus : config.stream_panda_buses) bus = normalize_live_panda_bus_config(bus);
  if (config.stream_source_kind == LiveSourceKind::SocketCan) {
    if (config.stream_address.empty()) config.stream_address = "vcan0";
  } else if (config.stream_source_kind == LiveSourceKind::PandaUsb) {
    // Empty serial means first available Panda.
  } else if (config.stream_source_kind == LiveSourceKind::CerealRemote ||
             config.stream_source_kind == LiveSourceKind::DeviceBridge) {
    config.stream_address = normalize_live_stream_address(std::move(config.stream_address));
  } else {
    config.stream_source_kind = live_is_local_stream_address(config.stream_address)
      ? LiveSourceKind::CerealLocal
      : LiveSourceKind::CerealRemote;
    config.stream_address = config.stream_source_kind == LiveSourceKind::CerealLocal
      ? "127.0.0.1"
      : normalize_live_stream_address(std::move(config.stream_address));
  }

  settings_path = config.settings_path.empty() ? default_loggy_settings_path() : fs::path(config.settings_path);
  LoggySettingsLoadResult loaded_settings = load_loggy_settings(settings_path);
  settings = std::move(loaded_settings.settings);
  if (!loaded_settings.error.empty()) {
    settings_status = "Settings load warning: " + loaded_settings.error;
  }
  for (const auto &[source_key, path] : settings.dbc_assignments) {
    SourceSet sources;
    std::string error;
    if (!parse_source_set(source_key, sources, error)) {
      settings_status = "Settings DBC skipped: " + source_key + " (" + error + ")";
      continue;
    }
    if (!dbc.open(sources, path, error)) {
      settings_status = "Settings DBC skipped: " + path + " (" + error + ")";
    }
  }
  manual_dbc_name = settings.dbc_override;
  if (!manual_dbc_name.empty()) {
    std::string ignore;
    apply_dbc_selection(ignore);
  }

  workspace_layout_path = resolve_layout_path(config.layout);
  if (workspace_layout_path.empty()) workspace_layout_path = resolve_preset_layout_path(config.preset);
  if (workspace_layout_path.empty()) {
    workspace = make_default_workspace(config.preset);
  } else {
    const WorkspaceLoadResult loaded_workspace = load_workspace_or_draft(workspace_layout_path);
    workspace = loaded_workspace.workspace;
    loaded_workspace_draft = loaded_workspace.loaded_draft;
  }
  normalize_workspace(&workspace);
  computed_specs = normalize_workspace_computed_series(&workspace);
  computed_dirty_ = !computed_specs.empty();

  const TimeRange initial_range = config.stream ? TimeRange{0.0, 30.0} : TimeRange{0.0, 60.0};
  playback.set_route_range(initial_range);
  playback.seek(initial_range.start_);
  view_range.set_route_range(initial_range);
  view_range.reset_to_route();
  timeline.set_route_range(initial_range);
  if (!config.route_name.empty() && !config.stream) {
    RouteIngestConfig ingest;
    ingest.resolve.route_name = config.route_name;
    ingest.resolve.data_dir = config.data_dir;
    ingest.worker_count = 2;
    route_ingest_.start(std::move(ingest));
  } else if (config.stream) {
    seed_demo_data();
    live_camera_source.set_enabled(true);
    live_poller_.start(live_source_config_from_session(config));
  }
}

LivePollSnapshot Session::live_status() const {
  LivePollSnapshot snapshot = live_poller_.snapshot();
  if (snapshot.error.empty() && !live_error_.empty()) snapshot.error = live_error_;
  return snapshot;
}

void Session::set_live_follow(bool follow) {
  live_follow = follow;
  if (!live_follow || !live_range_.valid()) return;
  const TimeRange display_range = live_display_range(live_range_, config.stream_buffer_seconds);
  playback.set_route_range(display_range);
  view_range.set_route_range(display_range);
  view_range.set_range(display_range);
  timeline.set_route_range(display_range);
  playback.seek(live_range_.end);
}

bool Session::toggle_live_follow() {
  set_live_follow(!live_follow);
  return live_follow;
}

void Session::set_live_paused(bool paused) {
  live_poller_.set_paused(paused);
}

bool Session::toggle_live_paused() {
  const bool paused = !live_paused();
  set_live_paused(paused);
  return paused;
}

bool Session::restart_live(std::string address, double buffer_seconds, std::string &error) {
  const LiveSourceKind kind = live_is_local_stream_address(address)
    ? LiveSourceKind::CerealLocal
    : LiveSourceKind::CerealRemote;
  return restart_live(kind, std::move(address), buffer_seconds, error);
}

bool Session::restart_live(LiveSourceKind source_kind, std::string address, double buffer_seconds, std::string &error) {
  LiveSourceConfig source = live_source_config_from_session(config);
  source.kind = source_kind;
  source.address = std::move(address);
  source.buffer_seconds = buffer_seconds;
  return restart_live(std::move(source), error);
}

bool Session::restart_live(LiveSourceConfig source, std::string &error) {
  if (!config.route_name.empty()) {
    error = "live source is unavailable while a route is open";
    return false;
  }
  source.address = trim_copy(std::move(source.address));
  for (PandaBusConfig &bus : source.panda_buses) bus = normalize_live_panda_bus_config(bus);
  if (source.kind == LiveSourceKind::SocketCan && source.address.empty()) {
    error = "SocketCAN device is empty";
    return false;
  }
  config.stream = true;
  config.stream_source_kind = source.kind;
  config.stream_address = (source.kind == LiveSourceKind::SocketCan || source.kind == LiveSourceKind::PandaUsb)
    ? std::move(source.address)
    : normalize_live_stream_address(std::move(source.address));
  config.stream_panda_buses = source.panda_buses;
  config.stream_buffer_seconds = std::max(1.0, source.buffer_seconds);
  live_error_.clear();
  live_range_ = {};
  live_follow = true;
  car_fingerprint.clear();
  auto_dbc_name.clear();
  if (manual_dbc_name.empty()) {
    std::string ignore;
    apply_dbc_selection(ignore);
  }
  route_timeline_spans_.clear();
  logs.clear();

  const TimeRange initial_range{0.0, config.stream_buffer_seconds};
  playback.set_route_range(initial_range);
  playback.seek(initial_range.start_);
  view_range.set_route_range(initial_range);
  view_range.reset_to_route();
  timeline.set_route_range(initial_range);
  timeline.clear_spans();

  seed_demo_data();
  live_camera_source.set_enabled(false);
  live_camera_source.set_enabled(true);
  live_poller_.start(live_source_config_from_session(config));
  error.clear();
  return true;
}

void Session::stop_live() {
  live_poller_.stop();
  live_camera_source.set_enabled(false);
  config.stream = false;
  live_error_.clear();
}

bool Session::restart_route(std::string route_name, std::string &error) {
  route_name = trim_copy(std::move(route_name));
  if (route_name.empty()) {
    error = "route name is empty";
    return false;
  }
  RouteSelection selection = parse_route_selection(route_name);
  if (selection.timestamp.empty()) {
    error = "invalid route format";
    return false;
  }

  live_poller_.stop();
  live_camera_source.set_enabled(false);
  route_ingest_.stop();
  config.stream = false;
  config.route_name = std::move(route_name);
  live_range_ = {};
  live_error_.clear();
  car_fingerprint.clear();
  auto_dbc_name.clear();
  if (manual_dbc_name.empty()) {
    std::string ignore;
    apply_dbc_selection(ignore);
  }
  route_timeline_spans_.clear();
  logs.clear();
  store.clear();
  scheduler.set_route_segments({});
  route_range_applied_ = false;
  camera_indexes_dirty_ = true;
  camera_index_segment_count_ = 0;
  computed_dirty_ = !computed_specs.empty();
  computed_statuses.clear();

  const TimeRange initial_range{0.0, 60.0};
  playback.set_route_range(initial_range);
  playback.seek(initial_range.start_);
  view_range.set_route_range(initial_range);
  view_range.reset_to_route();
  timeline.set_route_range(initial_range);
  timeline.clear_spans();

  RouteIngestConfig ingest;
  ingest.resolve.route_name = config.route_name;
  ingest.resolve.data_dir = config.data_dir;
  ingest.worker_count = 2;
    route_ingest_.start(std::move(ingest));
  error.clear();
  return true;
}

void Session::update_car_fingerprint(std::string fingerprint) {
  fingerprint = trim_copy(std::move(fingerprint));
  if (fingerprint.empty() || fingerprint == car_fingerprint) return;
  car_fingerprint = std::move(fingerprint);
  auto_dbc_name = dbc_name_for_fingerprint(car_fingerprint);
  std::string ignore;
  apply_dbc_selection(ignore);
}

bool Session::apply_dbc_selection(std::string &error) {
  const bool manual = !manual_dbc_name.empty();
  const std::string &desired = manual ? manual_dbc_name : auto_dbc_name;
  if (desired.empty()) {
    if (!active_dbc_name.empty()) dbc.close(SOURCE_ALL);
    active_dbc_name.clear();
    if (!manual && !car_fingerprint.empty()) {
      dbc_status = "No DBC mapping for " + car_fingerprint;
    } else {
      dbc_status.clear();
    }
    error.clear();
    return false;
  }
  if (desired == active_dbc_name && dbc.find_dbc_file(0) != nullptr) {
    error.clear();
    return true;
  }

  const std::optional<fs::path> path = resolve_dbc_reference(desired, settings);
  if (!path.has_value()) {
    dbc_status = "DBC not found: " + desired;
    error = dbc_status;
    return false;
  }

  std::string open_error;
  if (!dbc.open(SOURCE_ALL, path->string(), open_error)) {
    dbc_status = "DBC open failed: " + open_error;
    error = dbc_status;
    return false;
  }

  active_dbc_name = desired;
  remember_recent_dbc_file(&settings, path->string());
  dbc_status = manual ? "DBC override: " + desired : "DBC auto: " + desired;
  error.clear();
  return true;
}

bool Session::set_manual_dbc_name(std::string dbc_name, std::string &error) {
  const std::string previous_manual = manual_dbc_name;
  const std::string previous_settings_override = settings.dbc_override;
  manual_dbc_name = trim_copy(std::move(dbc_name));
  settings.dbc_override = manual_dbc_name;
  const bool applied = apply_dbc_selection(error);
  if (!applied && !manual_dbc_name.empty()) {
    manual_dbc_name = previous_manual;
    settings.dbc_override = previous_settings_override;
    std::string ignore;
    apply_dbc_selection(ignore);
    return false;
  }
  std::string save_error;
  if (!save_settings(save_error) && !save_error.empty()) {
    manual_dbc_name = previous_manual;
    settings.dbc_override = previous_settings_override;
    std::string ignore;
    apply_dbc_selection(ignore);
    if (error.empty()) error = save_error;
    return false;
  }
  return applied || manual_dbc_name.empty();
}

bool Session::save_settings(std::string &error) {
  std::string local_error;
  const bool saved = save_loggy_settings(settings, settings_path, local_error);
  if (saved) {
    settings_status = "Saved settings";
  } else {
    settings_status = "Settings save failed: " + local_error;
  }
  error = local_error;
  return saved;
}

SelectionContext &Session::selection(std::string_view group) {
  const std::string key(group.empty() ? "default" : group);
  for (auto &[name, context] : selections_) {
    if (name == key) return context;
  }
  selections_.push_back({key, SelectionContext{}});
  return selections_.back().second;
}

DrainResult Session::begin_frame() {
  scheduler.set_tracker_time(playback.tracker_time());
  scheduler.set_visible_ranges({view_range.range()});

  const RouteIngestStatus status = route_ingest_.status();
  if (!status.car_fingerprint.empty()) update_car_fingerprint(status.car_fingerprint);
  if (!route_range_applied_ && status.route_range.valid() && status.route_range.span() > 0.0) {
    route_range_applied_ = true;
    playback.set_route_range(status.route_range);
    playback.seek(status.route_range.start_);
    view_range.set_route_range(status.route_range);
    view_range.reset_to_route();
    timeline.set_route_range(status.route_range);
  }

  std::vector<TimelineSpan> new_spans = route_ingest_.drain_timeline_spans();
  if (!new_spans.empty()) {
    route_timeline_spans_.insert(route_timeline_spans_.end(),
                                 std::make_move_iterator(new_spans.begin()),
                                 std::make_move_iterator(new_spans.end()));
    route_timeline_spans_ = merge_timeline_spans(std::move(route_timeline_spans_));
    timeline.set_spans(route_timeline_spans_);
  }

  append_and_sort_logs(&logs, route_ingest_.drain_log_entries());

  TimeRange live_keep_range;
  if (config.stream) {
    LiveExtractBatch live_batch;
    const LivePollResult consume_result = live_poller_.consume(live_batch);
    if (consume_result.has_update) {
      if (!consume_result.error.empty()) live_error_ = consume_result.error;
      if (consume_result.has_batch) {
        if (!live_batch.car_fingerprint.empty()) update_car_fingerprint(live_batch.car_fingerprint);
        if (live_batch.range.valid()) {
          const TimeRange previous_live_range = live_range_;
          extend_range(&live_range_, live_batch.range);
          const double latest = live_range_.end;
          const TimeRange display_range = live_display_range(live_range_, config.stream_buffer_seconds);
          live_keep_range = display_range;
          const bool first_live_batch = !previous_live_range.valid();
          playback.set_route_range(display_range);
          view_range.set_route_range(display_range);
          timeline.set_route_range(display_range);
          if (first_live_batch || live_follow) {
            playback.seek(latest);
            view_range.set_range(display_range);
          }
        }
        if (!live_batch.timeline_spans.empty()) {
          route_timeline_spans_.insert(route_timeline_spans_.end(),
                                       std::make_move_iterator(live_batch.timeline_spans.begin()),
                                       std::make_move_iterator(live_batch.timeline_spans.end()));
          route_timeline_spans_ = merge_timeline_spans(std::move(route_timeline_spans_));
          timeline.set_spans(route_timeline_spans_);
        }
        append_and_sort_logs(&logs, std::move(live_batch.logs));
        if (live_batch.has_time_offset && live_batch.store.coverage.empty() && live_batch.range.valid()) {
          live_batch.store.coverage.push_back(live_batch.range);
        }
        if (!live_batch.store.series.empty() || !live_batch.store.can_events.empty()) {
          store.stage(std::move(live_batch.store));
        }
      }
    }
  }

  DrainResult drain = store.begin_frame();
  if (config.stream && live_keep_range.valid() && live_keep_range.start_ > 0.0) {
    const StoreTrimResult trim = store.trim_before(live_keep_range.start_);
    merge_trim_result(&drain, trim);
    trim_timeline_spans_before(&route_timeline_spans_, live_keep_range.start_);
    trim_logs_before(&logs, live_keep_range.start_);
    timeline.set_spans(route_timeline_spans_);
  }

  const std::vector<RouteSegment> route_segments = scheduler.segments();
  if (route_segments.size() != camera_index_segment_count_) {
    camera_index_segment_count_ = route_segments.size();
    camera_indexes_dirty_ = true;
  }
  if (camera_indexes_dirty_ || camera_dependencies_touched(drain.touched_series_paths)) {
    TimeRange range = playback.route_range();
    if (!range.valid() || range.span() <= 0.0) range = view_range.range();
    refresh_camera_indexes(route_segments, store, range, &camera_indexes_);
    for (const CameraViewSpec &spec : camera_view_specs()) {
      const size_t index = camera_view_index(spec.view);
      camera_decoders_[index].set_camera_index(camera_indexes_[index]);
    }
    camera_indexes_dirty_ = false;
  }

  if (!computed_specs.empty() &&
      (computed_dirty_ || computed_dependencies_touched(computed_specs, drain.touched_series_paths))) {
    TimeRange range = playback.route_range();
    if (!range.valid() || range.span() <= 0.0) range = view_range.range();
    computed_statuses.clear();
    StoreBatch computed_batch = materialize_computed_series_batch(store, computed_specs, range, &computed_statuses);
    computed_dirty_ = false;
    if (!computed_batch.replace_series_paths.empty() || !computed_batch.series.empty()) {
      store.stage(std::move(computed_batch));
      merge_drain_result(&drain, store.begin_frame());
    }
  }

  return drain;
}

void Session::seed_demo_data() {
  if (demo_seeded_) return;
  demo_seeded_ = true;

  constexpr double kEnd = 60.0;
  StoreBatch batch;
  batch.segment = 0;
  batch.coverage = {{0.0, kEnd}};

  SeriesChunk v_ego;
  v_ego.path = "/carState/vEgo";
  v_ego.range = {0.0, kEnd};
  v_ego.segment = 0;

  SeriesChunk a_ego;
  a_ego.path = "/carState/aEgo";
  a_ego.range = {0.0, kEnd};
  a_ego.segment = 0;

  SeriesChunk gps_lat;
  gps_lat.path = "/gpsLocationExternal/latitude";
  gps_lat.range = {0.0, kEnd};
  gps_lat.segment = 0;

  SeriesChunk gps_lon;
  gps_lon.path = "/gpsLocationExternal/longitude";
  gps_lon.range = {0.0, kEnd};
  gps_lon.segment = 0;

  SeriesChunk gps_fix;
  gps_fix.path = "/gpsLocationExternal/hasFix";
  gps_fix.range = {0.0, kEnd};
  gps_fix.segment = 0;

  SeriesChunk gps_bearing;
  gps_bearing.path = "/gpsLocationExternal/bearingDeg";
  gps_bearing.range = {0.0, kEnd};
  gps_bearing.segment = 0;

  auto make_encode_series = [&](std::string path, bool constant_zero) {
    SeriesChunk chunk;
    chunk.path = std::move(path);
    chunk.range = {0.0, 30.0};
    chunk.segment = 0;
    for (int i = 0; i <= 300; ++i) {
      chunk.points.push_back({static_cast<double>(i) * 0.1, constant_zero ? 0.0 : static_cast<double>(i)});
    }
    return chunk;
  };

  for (int i = 0; i <= 600; ++i) {
    const double t = static_cast<double>(i) * 0.1;
    const double v = 18.0 + 4.0 * std::sin(t * 0.28);
    v_ego.points.push_back({t, v});
    a_ego.points.push_back({t, 1.1 * std::cos(t * 0.28)});
    gps_lat.points.push_back({t, 37.0000 + 0.0012 * std::sin(t * 0.045)});
    gps_lon.points.push_back({t, -122.0000 + 0.0016 * std::cos(t * 0.045)});
    gps_fix.points.push_back({t, 1.0});
    gps_bearing.points.push_back({t, std::fmod(90.0 + t * 2.0, 360.0)});
  }
  batch.series.push_back(std::move(v_ego));
  batch.series.push_back(std::move(a_ego));
  batch.series.push_back(std::move(gps_lat));
  batch.series.push_back(std::move(gps_lon));
  batch.series.push_back(std::move(gps_fix));
  batch.series.push_back(std::move(gps_bearing));
  for (const CameraViewSpec &spec : camera_view_specs()) {
    const std::string prefix = "/" + std::string(spec.encode_index);
    batch.series.push_back(make_encode_series(prefix + "/segmentNum", true));
    batch.series.push_back(make_encode_series(prefix + "/segmentId", false));
    batch.series.push_back(make_encode_series(prefix + "/frameId", false));
  }

  CanEventChunk can;
  can.id = MessageId{.source = 0, .address = 0x123};
  can.range = {0.0, kEnd};
  can.segment = 0;
  for (int i = 0; i <= 120; ++i) {
    const double t = static_cast<double>(i) * 0.5;
    const uint8_t b0 = static_cast<uint8_t>(i & 0xff);
    const uint8_t b1 = static_cast<uint8_t>((i * 3) & 0xff);
    can.events.push_back({.mono_time = t, .bus_time = static_cast<uint16_t>(i), .data = {b0, b1, 0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc}});
  }
  batch.can_events.push_back(std::move(can));
  store.stage(std::move(batch));

  scheduler.set_route_segments({
    RouteSegment{
      .segment = 0,
      .range = {0.0, kEnd},
      .log_path = "demo.qlog",
      .road_camera_path = "/tmp/loggy_demo_fcamera.hevc",
      .driver_camera_path = "/tmp/loggy_demo_dcamera.hevc",
      .wide_road_camera_path = "/tmp/loggy_demo_ecamera.hevc",
      .qroad_camera_path = "/tmp/loggy_demo_qcamera.ts",
      .state = SegmentState::Loaded,
    },
  });
  camera_indexes_dirty_ = true;

  timeline.set_spans(
    std::vector<EngagementSpan>{{5.0, 20.0}, {32.0, 44.0}},
    std::vector<AlertSpan>{{18.0, 21.0, AlertLevel::Warning}, {43.0, 46.0, AlertLevel::Critical}});
}

}  // namespace loggy
