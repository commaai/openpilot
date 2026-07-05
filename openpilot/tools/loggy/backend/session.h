#pragma once

#include "tools/loggy/backend/dbc/dbc.h"
#include "tools/loggy/backend/ingest.h"
#include "tools/loggy/backend/route.h"
#include "tools/loggy/backend/store.h"
#include "tools/loggy/shell/settings.h"
#include "tools/loggy/shell/transport.h"
#include "tools/loggy/shell/workspace.h"

#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace loggy {

struct SessionConfig {
  std::string preset = "loggy";
  std::string layout;
  std::string route_name;
  std::string data_dir;
  std::string settings_path;
  std::string stream_address = "127.0.0.1";
  bool stream = false;
};

struct SelectionContext {
  MessageId selected_msg_id;
  bool has_selected_msg = false;
};

class Session {
public:
  explicit Session(SessionConfig config);

  const SessionConfig &config() const { return config_; }
  Workspace &workspace() { return workspace_; }
  const Workspace &workspace() const { return workspace_; }
  PlaybackClock &playback() { return playback_; }
  const PlaybackClock &playback() const { return playback_; }
  SharedViewRange &view_range() { return view_range_; }
  const SharedViewRange &view_range() const { return view_range_; }
  TimelineModel &timeline() { return timeline_; }
  const TimelineModel &timeline() const { return timeline_; }
  Store &store() { return store_; }
  const Store &store() const { return store_; }
  SegmentScheduler &scheduler() { return scheduler_; }
  const SegmentScheduler &scheduler() const { return scheduler_; }
  RouteIngestStatus ingestStatus() const { return route_ingest_.status(); }
  const std::vector<LogEntry> &logs() const { return route_logs_; }
  LoggySettings &settings() { return settings_; }
  const LoggySettings &settings() const { return settings_; }
  const std::filesystem::path &settings_path() const { return settings_path_; }
  const std::string &settings_status() const { return settings_status_; }
  bool saveSettings(std::string *error = nullptr);
  SelectionContext &selection(std::string_view group);
  DrainResult beginFrame();
  void seedDemoData();

private:
  SessionConfig config_;
  Workspace workspace_;
  PlaybackClock playback_;
  SharedViewRange view_range_;
  TimelineModel timeline_;
  Store store_;
  SegmentScheduler scheduler_;
  RouteIngestor route_ingest_;
  LoggySettings settings_;
  std::filesystem::path settings_path_;
  std::string settings_status_;
  std::vector<TimelineSpan> route_timeline_spans_;
  std::vector<LogEntry> route_logs_;
  bool route_range_applied_ = false;
  bool demo_seeded_ = false;
  std::vector<std::pair<std::string, SelectionContext>> selections_;
};

}  // namespace loggy
