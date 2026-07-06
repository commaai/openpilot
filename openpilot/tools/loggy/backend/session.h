#pragma once

#include "tools/loggy/backend/dbc/dbc.h"
#include "tools/loggy/backend/computed.h"
#include "tools/loggy/backend/ingest.h"
#include "tools/loggy/backend/live.h"
#include "tools/loggy/backend/route.h"
#include "tools/loggy/backend/store.h"
#include "tools/loggy/backend/dbc/undo.h"
#include "tools/loggy/backend/video.h"
#include "tools/loggy/shell/settings.h"
#include "tools/loggy/shell/transport.h"
#include "tools/loggy/shell/workspace.h"

#include <filesystem>
#include <future>
#include <array>
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
  LiveSourceKind stream_source_kind = LiveSourceKind::CerealLocal;
  std::array<PandaBusConfig, kPandaBusCount> stream_panda_buses{};
  double stream_buffer_seconds = 30.0;
  bool stream = false;
};

struct SelectionContext {
  MessageId selected_msg_id;
  bool has_selected_msg = false;
};

class Session {
public:
  explicit Session(SessionConfig cfg);

  SessionConfig config;
  std::filesystem::path workspace_layout_path;
  bool loaded_workspace_draft = false;
  Workspace workspace;
  PlaybackClock playback;
  SharedViewRange view_range;
  TimelineModel timeline;
  Store store;
  DBCManager dbc;
  UndoStack dbc_undo;
  // One-shot request from the Browser (double-click a series): the first plot pane drawn after
  // it is set appends the path. Aged out after two frames so it can't fire on a later tab
  // switch if the current tab happens to have no plot pane.
  std::string pending_plot_series;
  int pending_plot_series_age = 0;
  SegmentScheduler scheduler;
  LoggySettings settings;
  std::filesystem::path settings_path;
  std::string settings_status;
  std::vector<LogEntry> logs;
  std::string car_fingerprint;
  std::string auto_dbc_name;
  std::string manual_dbc_name;
  std::string active_dbc_name;
  std::string dbc_status;
  bool live_follow = true;
  LiveCameraFrameSource live_camera_source;
  std::vector<ComputedSeriesSpec> computed_specs;
  std::vector<ComputedSeriesStatus> computed_statuses;

  RouteIngestStatus ingest_status() const { return route_ingest_.status(); }
  LivePollSnapshot live_status() const;
  void set_live_follow(bool follow);
  bool toggle_live_follow();
  bool live_paused() const { return live_status().paused; }
  void set_live_paused(bool paused);
  bool toggle_live_paused();
  bool restart_live(std::string address, double buffer_seconds, std::string &error);
  bool restart_live(LiveSourceKind source_kind, std::string address, double buffer_seconds, std::string &error);
  bool restart_live(LiveSourceConfig source, std::string &error);
  void stop_live();
  bool restart_route(std::string route_name, std::string &error);
  bool set_manual_dbc_name(std::string dbc_name, std::string &error);
  const CameraFeedIndex &camera_index(CameraViewKind view) const { return camera_indexes_[camera_view_index(view)]; }
  CameraFrameDecoder &camera_decoder(CameraViewKind view) { return camera_decoders_[camera_view_index(view)]; }
  bool save_settings(std::string &error);
  SelectionContext &selection(std::string_view group);
  DrainResult begin_frame();
  void seed_demo_data();

private:
  void update_car_fingerprint(std::string fingerprint);
  bool apply_dbc_selection(std::string &error);

  RouteIngestor route_ingest_;
  // In-flight auto DBC parse (fingerprint arrival mid-load): parsed on a worker, adopted in
  // begin_frame — parsing ~10k-line opendbc files on the UI thread dropped a frame.
  std::future<std::pair<std::string, std::shared_ptr<DBCFile>>> pending_dbc_load_;
  LiveCerealPoller live_poller_;
  std::vector<TimelineSpan> route_timeline_spans_;
  TimeRange live_range_;
  std::string live_error_;
  std::array<CameraFeedIndex, 4> camera_indexes_;
  std::array<CameraFrameDecoder, 4> camera_decoders_;
  bool camera_indexes_dirty_ = true;
  // Coarse-cadence rebuild state for the camera frame indexes (see begin_frame).
  bool camera_indexes_stale_ = false;
  double camera_indexes_built_at_ = -1.0e9;
  size_t camera_index_segment_count_ = 0;
  bool computed_dirty_ = false;
  bool route_range_applied_ = false;
  // Latches once ingest completes and route_range has been snapped from the nominal
  // 60s/segment span down to the real content end (REVIEW.md defect #34).
  bool real_range_applied_ = false;
  bool demo_seeded_ = false;
  std::vector<std::pair<std::string, SelectionContext>> selections_;
};

}  // namespace loggy
