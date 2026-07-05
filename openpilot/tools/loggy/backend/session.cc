#include "tools/loggy/backend/session.h"

#include "tools/loggy/backend/dbc/dbcmanager.h"

#include <filesystem>
#include <algorithm>
#include <cmath>
#include <iterator>
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

}  // namespace

Session::Session(SessionConfig config) : config_(std::move(config)), scheduler_(&store_), route_ingest_(&scheduler_) {
  register_dummy_pane_types();

  settings_path_ = config_.settings_path.empty() ? default_loggy_settings_path() : fs::path(config_.settings_path);
  LoggySettingsLoadResult loaded_settings = load_loggy_settings(settings_path_);
  settings_ = std::move(loaded_settings.settings);
  if (!loaded_settings.error.empty()) {
    settings_status_ = "Settings load warning: " + loaded_settings.error;
  }
  for (const auto &[source_key, path] : settings_.dbc_assignments) {
    SourceSet sources;
    std::string error;
    if (!parseSourceSet(source_key, &sources, &error)) {
      settings_status_ = "Settings DBC skipped: " + source_key + " (" + error + ")";
      continue;
    }
    if (!dbc()->open(sources, path, &error)) {
      settings_status_ = "Settings DBC skipped: " + path + " (" + error + ")";
    }
  }

  fs::path layout_path = resolve_layout_path(config_.layout);
  if (layout_path.empty()) layout_path = resolve_preset_layout_path(config_.preset);
  workspace_ = layout_path.empty() ? make_default_workspace(config_.preset) : load_workspace_json(layout_path);
  normalize_workspace(&workspace_);

  const TimeRange initial_range = config_.stream ? TimeRange{0.0, 30.0} : TimeRange{0.0, 60.0};
  playback_.set_route_range(initial_range);
  playback_.seek(initial_range.start);
  view_range_.set_route_range(initial_range);
  view_range_.reset_to_route();
  timeline_.set_route_range(initial_range);
  if (!config_.route_name.empty() && !config_.stream) {
    RouteIngestConfig ingest;
    ingest.resolve.route_name = config_.route_name;
    ingest.resolve.data_dir = config_.data_dir;
    ingest.worker_count = 2;
    route_ingest_.start(std::move(ingest));
  } else if (config_.stream) {
    seedDemoData();
  }
}

bool Session::saveSettings(std::string *error) {
  std::string local_error;
  const bool saved = save_loggy_settings(settings_, settings_path_, &local_error);
  if (saved) {
    settings_status_ = "Saved settings";
  } else {
    settings_status_ = "Settings save failed: " + local_error;
  }
  if (error != nullptr) *error = local_error;
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

DrainResult Session::beginFrame() {
  scheduler_.setTrackerTime(playback_.tracker_time());
  scheduler_.setVisibleRanges({view_range_.range()});

  const RouteIngestStatus status = route_ingest_.status();
  if (!route_range_applied_ && status.route_range.valid() && status.route_range.span() > 0.0) {
    route_range_applied_ = true;
    playback_.set_route_range(status.route_range);
    playback_.seek(status.route_range.start);
    view_range_.set_route_range(status.route_range);
    view_range_.reset_to_route();
    timeline_.set_route_range(status.route_range);
  }

  std::vector<TimelineSpan> new_spans = route_ingest_.drainTimelineSpans();
  if (!new_spans.empty()) {
    route_timeline_spans_.insert(route_timeline_spans_.end(),
                                 std::make_move_iterator(new_spans.begin()),
                                 std::make_move_iterator(new_spans.end()));
    route_timeline_spans_ = merge_timeline_spans(std::move(route_timeline_spans_));
    timeline_.set_spans(route_timeline_spans_);
  }

  append_and_sort_logs(&route_logs_, route_ingest_.drainLogEntries());

  return store_.beginFrame();
}

void Session::seedDemoData() {
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

  for (int i = 0; i <= 600; ++i) {
    const double t = static_cast<double>(i) * 0.1;
    const double v = 18.0 + 4.0 * std::sin(t * 0.28);
    v_ego.points.push_back({t, v});
    a_ego.points.push_back({t, 1.1 * std::cos(t * 0.28)});
  }
  batch.series.push_back(std::move(v_ego));
  batch.series.push_back(std::move(a_ego));

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
  store_.stage(std::move(batch));

  timeline_.set_spans(
    std::vector<EngagementSpan>{{5.0, 20.0}, {32.0, 44.0}},
    std::vector<AlertSpan>{{18.0, 21.0, AlertLevel::Warning}, {43.0, 46.0, AlertLevel::Critical}});
}

}  // namespace loggy
