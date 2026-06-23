#include "tools/jotpluggler/internal.h"

#include <tuple>

template <typename Cmp, typename SeriesAccessor, typename LogAccessor>
std::optional<double> stream_batch_extreme_time(const StreamExtractBatch &batch,
                                                Cmp cmp,
                                                SeriesAccessor series_time,
                                                LogAccessor log_time_fn) {
  std::optional<double> result;
  for (const RouteSeries &series : batch.series) {
    if (!series.times.empty()) {
      const double t = series_time(series);
      result = result.has_value() ? cmp(*result, t) : t;
    }
  }
  if (!batch.logs.empty()) {
    const double t = log_time_fn(batch);
    result = result.has_value() ? cmp(*result, t) : t;
  }
  if (!batch.timeline.empty()) {
    const double t = cmp(batch.timeline.front().start_time, batch.timeline.back().end_time);
    result = result.has_value() ? cmp(*result, t) : t;
  }
  for (const CanMessageData &message : batch.can_messages) {
    if (!message.samples.empty()) {
      const double t = cmp(message.samples.front().mono_time, message.samples.back().mono_time);
      result = result.has_value() ? cmp(*result, t) : t;
    }
  }
  return result;
}

std::optional<double> earliest_stream_batch_time(const StreamExtractBatch &batch) {
  return stream_batch_extreme_time(batch,
    [](double a, double b) { return std::min(a, b); },
    [](const RouteSeries &s) { return s.times.front(); },
    [](const StreamExtractBatch &b) { return b.logs.front().mono_time; });
}

std::optional<double> latest_stream_batch_time(const StreamExtractBatch &batch) {
  return stream_batch_extreme_time(batch,
    [](double a, double b) { return std::max(a, b); },
    [](const RouteSeries &s) { return s.times.back(); },
    [](const StreamExtractBatch &b) { return b.logs.back().mono_time; });
}

bool layout_has_custom_curves(const SketchLayout &layout) {
  for (const WorkspaceTab &tab : layout.tabs) {
    for (const Pane &pane : tab.panes) {
      for (const Curve &curve : pane.curves) {
        if (curve.custom_python.has_value()) return true;
      }
    }
  }
  return false;
}

void append_stream_timeline_entries(std::vector<TimelineEntry> *timeline, std::vector<TimelineEntry> entries) {
  for (TimelineEntry &entry : entries) {
    if (!timeline->empty() && timeline->back().type == entry.type) {
      timeline->back().end_time = std::max(timeline->back().end_time, entry.end_time);
    } else {
      timeline->push_back(std::move(entry));
    }
  }
}

bool can_message_less(const CanMessageData &a, const CanMessageData &b) {
  return std::make_tuple(a.id.service, a.id.bus, a.id.address)
       < std::make_tuple(b.id.service, b.id.bus, b.id.address);
}

void apply_stream_batch(AppSession *session, UiState *state, StreamExtractBatch batch) {
  if (batch.has_time_offset) {
    session->stream_time_offset = batch.time_offset;
  }
  if (!batch.car_fingerprint.empty()) {
    session->route_data.car_fingerprint = batch.car_fingerprint;
  }
  if (!batch.dbc_name.empty()) {
    session->route_data.dbc_name = batch.dbc_name;
  }
  if (!batch.enum_info.empty()) {
    for (auto &[path, info] : batch.enum_info) {
      session->route_data.enum_info[path] = std::move(info);
    }
  }

  bool new_paths = false;
  std::vector<RouteSeries> new_series;
  std::vector<std::string> touched_paths;
  touched_paths.reserve(batch.series.size());
  for (RouteSeries &incoming : batch.series) {
    touched_paths.push_back(incoming.path);
    auto existing_it = session->series_by_path.find(incoming.path);
    if (existing_it == session->series_by_path.end()) {
      new_series.push_back(std::move(incoming));
      new_paths = true;
      continue;
    }
    RouteSeries &existing = *existing_it->second;
    existing.times.insert(existing.times.end(), incoming.times.begin(), incoming.times.end());
    existing.values.insert(existing.values.end(), incoming.values.begin(), incoming.values.end());
  }
  for (RouteSeries &series : new_series) {
    session->route_data.paths.push_back(series.path);
    session->route_data.series.push_back(std::move(series));
  }

  if (!batch.logs.empty()) {
    std::sort(batch.logs.begin(), batch.logs.end(), [](const LogEntry &a, const LogEntry &b) {
      return a.mono_time < b.mono_time;
    });
    const size_t old_size = session->route_data.logs.size();
    session->route_data.logs.insert(session->route_data.logs.end(),
                                    std::make_move_iterator(batch.logs.begin()),
                                    std::make_move_iterator(batch.logs.end()));
    if (old_size > 0 && session->route_data.logs.size() > old_size
        && session->route_data.logs[old_size - 1].mono_time > session->route_data.logs[old_size].mono_time) {
      std::inplace_merge(session->route_data.logs.begin(),
                         session->route_data.logs.begin() + static_cast<ptrdiff_t>(old_size),
                         session->route_data.logs.end(),
                         [](const LogEntry &a, const LogEntry &b) {
                           return a.mono_time < b.mono_time;
                         });
    }
  }
  if (!batch.timeline.empty()) {
    append_stream_timeline_entries(&session->route_data.timeline, std::move(batch.timeline));
  }

  for (CanMessageData &incoming : batch.can_messages) {
    auto it = std::lower_bound(session->route_data.can_messages.begin(),
                               session->route_data.can_messages.end(),
                               incoming,
                               can_message_less);
    if (it == session->route_data.can_messages.end()
        || can_message_less(incoming, *it)
        || can_message_less(*it, incoming)) {
      session->route_data.can_messages.insert(it, std::move(incoming));
    } else {
      it->samples.insert(it->samples.end(),
                         std::make_move_iterator(incoming.samples.begin()),
                         std::make_move_iterator(incoming.samples.end()));
    }
  }

  if (new_paths) {
    const size_t old_path_count = session->route_data.paths.size() - new_series.size();
    std::sort(session->route_data.paths.begin() + static_cast<ptrdiff_t>(old_path_count), session->route_data.paths.end());
    std::inplace_merge(session->route_data.paths.begin(),
                       session->route_data.paths.begin() + static_cast<ptrdiff_t>(old_path_count),
                       session->route_data.paths.end());
    const size_t old_series_count = session->route_data.series.size() - new_series.size();
    auto series_cmp = [](const RouteSeries &a, const RouteSeries &b) { return a.path < b.path; };
    std::sort(session->route_data.series.begin() + static_cast<ptrdiff_t>(old_series_count),
              session->route_data.series.end(), series_cmp);
    std::inplace_merge(session->route_data.series.begin(),
                       session->route_data.series.begin() + static_cast<ptrdiff_t>(old_series_count),
                       session->route_data.series.end(), series_cmp);
    session->route_data.roots = collect_route_roots_for_paths(session->route_data.paths);
    rebuild_route_index(session);
    rebuild_browser_nodes(session, state);
    state->browser_nodes_dirty = false;
  } else {
    for (const std::string &path : touched_paths) {
      auto series_it = session->series_by_path.find(path);
      if (series_it == session->series_by_path.end() || series_it->second == nullptr) continue;
      const bool enum_like = session->route_data.enum_info.find(path) != session->route_data.enum_info.end();
      session->route_data.series_formats[path] = compute_series_format(series_it->second->values, enum_like);
    }
  }
  const std::optional<double> earliest_time = earliest_stream_batch_time(batch);
  const std::optional<double> latest_time = latest_stream_batch_time(batch);
  if (earliest_time.has_value() && latest_time.has_value()) {
    if (!session->route_data.has_time_range) {
      session->route_data.x_min = *earliest_time;
      session->route_data.x_max = *latest_time;
    } else {
      session->route_data.x_min = std::min(session->route_data.x_min, *earliest_time);
      session->route_data.x_max = std::max(session->route_data.x_max, *latest_time);
    }
    session->route_data.has_time_range = true;
  }

  if (new_paths
      || std::find(touched_paths.begin(), touched_paths.end(), "/gpsLocationExternal/latitude") != touched_paths.end()
      || std::find(touched_paths.begin(), touched_paths.end(), "/gpsLocationExternal/longitude") != touched_paths.end()
      || std::find(touched_paths.begin(), touched_paths.end(), "/gpsLocationExternal/hasFix") != touched_paths.end()
      || std::find(touched_paths.begin(), touched_paths.end(), "/gpsLocationExternal/bearingDeg") != touched_paths.end()) {
    rebuild_gps_trace(&session->route_data);
  }

  if (latest_time.has_value() && layout_has_custom_curves(session->layout)
      && *latest_time >= session->next_stream_custom_refresh_time) {
    refresh_all_custom_curves(session, state);
    session->next_stream_custom_refresh_time = *latest_time + 0.1;
  }
  if (state->follow_latest || !state->has_tracker_time) {
    state->tracker_time = session->route_data.x_max;
    state->has_tracker_time = session->route_data.has_time_range;
  }
  if (!state->has_shared_range) {
    reset_shared_range(state, *session);
  }
}
