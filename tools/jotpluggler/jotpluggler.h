#pragma once

#include "cereal/gen/cpp/log.capnp.h"
#include "imgui.h"
#include "tools/jotpluggler/dbc_core.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <future>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

// *****
// app options & entry point
// *****

struct Options {
  std::string layout;
  std::string route_name;
  std::string data_dir;
  std::string output_path;
  std::string stream_address = "127.0.0.1";
  int width = 1600;
  int height = 900;
  bool show = false;
  bool sync_load = false;
  bool stream = false;
  bool start_cabana = false;
  double stream_buffer_seconds = 30.0;
};

int run(const Options &options);

// *****
// sketch layout & route data
// *****

struct PlotRange {
  bool valid = false;
  double left = 0.0;
  double right = 0.0;
  double bottom = 0.0;
  double top = 1.0;
  bool has_y_limit_min = false;
  bool has_y_limit_max = false;
  double y_limit_min = 0.0;
  double y_limit_max = 1.0;
};

struct CustomPythonSeries {
  std::string linked_source;
  std::vector<std::string> additional_sources;
  std::string globals_code;
  std::string function_code;
};

struct Curve {
  std::string name;
  std::string label;
  std::array<uint8_t, 3> color = {160, 170, 180};
  bool visible = true;
  bool derivative = false;
  double derivative_dt = 0.0;
  double value_scale = 1.0;
  double value_offset = 0.0;
  bool runtime_only = false;
  std::optional<CustomPythonSeries> custom_python;
  std::string runtime_error_message;
  std::vector<double> xs;
  std::vector<double> ys;
};

enum class PaneKind : uint8_t {
  Plot,
  Map,
  Camera,
};

enum class CameraViewKind : uint8_t {
  Road,
  Driver,
  WideRoad,
  QRoad,
};

struct Pane {
  PaneKind kind = PaneKind::Plot;
  CameraViewKind camera_view = CameraViewKind::Road;
  std::string title;
  PlotRange range;
  std::vector<Curve> curves;
};

enum class SplitOrientation {
  Horizontal,
  Vertical,
};

struct WorkspaceNode {
  bool is_pane = false;
  int pane_index = -1;
  SplitOrientation orientation = SplitOrientation::Horizontal;
  std::vector<float> sizes;
  std::vector<WorkspaceNode> children;
};

struct WorkspaceTab {
  std::string tab_name;
  WorkspaceNode root;
  std::vector<Pane> panes;
};

struct RouteSeries {
  std::string path;
  std::vector<double> times;
  std::vector<double> values;
};

struct CameraSegmentFile {
  int segment = -1;
  std::string path;
};

struct CameraFrameIndexEntry {
  double timestamp = 0.0;
  int segment = -1;
  int decode_index = -1;
  uint32_t frame_id = 0;
};

struct CameraFeedIndex {
  std::vector<CameraSegmentFile> segment_files;
  std::vector<CameraFrameIndexEntry> entries;
};

enum class LogOrigin : uint8_t {
  Log,
  Android,
  Alert,
};

struct LogEntry {
  double mono_time = 0.0;
  double boot_time = 0.0;
  double wall_time = 0.0;
  uint8_t level = 20;
  std::string source;
  std::string func;
  std::string message;
  std::string context;
  LogOrigin origin = LogOrigin::Log;
};

struct EnumInfo {
  std::vector<std::string> names;
};

struct SeriesFormat {
  int decimals = 3;
  bool integer_like = false;
  bool has_negative = false;
  int digits_before = 1;
  int total_width = 0;
  char fmt[16] = "%7.3f";
};

enum class CanServiceKind : uint8_t {
  Can,
  Sendcan,
};

struct CanMessageId {
  CanServiceKind service = CanServiceKind::Can;
  uint8_t bus = 0;
  uint32_t address = 0;

  bool operator==(const CanMessageId &other) const {
    return service == other.service && bus == other.bus && address == other.address;
  }
};

struct CanMessageIdHash {
  size_t operator()(const CanMessageId &id) const {
    return (static_cast<size_t>(id.service) << 40)
         ^ (static_cast<size_t>(id.bus) << 32)
         ^ static_cast<size_t>(id.address);
  }
};

struct CanFrameSample {
  double mono_time = 0.0;
  uint16_t bus_time = 0;
  std::string data;
};

struct LiveCanFrame {
  double mono_time = 0.0;
  uint8_t bus = 0;
  uint32_t address = 0;
  uint16_t bus_time = 0;
  std::string data;
};

struct CanMessageData {
  CanMessageId id;
  std::vector<CanFrameSample> samples;
};

struct TimelineEntry {
  enum class Type : uint8_t {
    None,
    Engaged,
    AlertInfo,
    AlertWarning,
    AlertCritical,
  };

  double start_time = 0.0;
  double end_time = 0.0;
  Type type = Type::None;
};

struct GpsPoint {
  double time = 0.0;
  double lat = 0.0;
  double lon = 0.0;
  float bearing = 0.0f;
  TimelineEntry::Type type = TimelineEntry::Type::None;
};

struct GpsTrace {
  std::vector<GpsPoint> points;
  double min_lat = 0.0;
  double max_lat = 0.0;
  double min_lon = 0.0;
  double max_lon = 0.0;
};

enum class LogSelector : uint8_t {
  Auto,
  RLog,
  QLog,
};

struct RouteIdentifier {
  std::string dongle_id;
  std::string log_id;
  int slice_begin = 0;
  int slice_end = -1;
  bool slice_explicit = false;
  LogSelector selector = LogSelector::Auto;
  bool selector_explicit = false;
  int available_begin = 0;
  int available_end = 0;

  bool empty() const {
    return dongle_id.empty() || log_id.empty();
  }

  std::string canonical() const {
    return empty() ? std::string() : dongle_id + "/" + log_id;
  }

  std::string onebox() const {
    return empty() ? std::string() : dongle_id + "|" + log_id;
  }

  std::string display_slice() const {
    const int begin = slice_explicit ? slice_begin : available_begin;
    const int end = slice_explicit ? slice_end : available_end;
    if (end < 0 || end == begin) {
      return std::to_string(begin);
    }
    return std::to_string(begin) + ":" + std::to_string(end);
  }

  char selector_char() const {
    switch (selector) {
      case LogSelector::RLog: return 'r';
      case LogSelector::QLog: return 'q';
      case LogSelector::Auto:
      default: return 'a';
    }
  }

  std::string full_spec() const {
    if (empty()) return {};
    std::string spec = dongle_id + "/" + log_id;
    if (slice_explicit) {
      spec += "/";
      spec += display_slice();
    }
    if (selector_explicit) {
      spec += "/";
      spec.push_back(selector_char());
    }
    return spec;
  }
};

struct RouteData {
  std::vector<RouteSeries> series;
  std::vector<std::string> paths;
  std::vector<std::string> roots;
  std::vector<CanMessageData> can_messages;
  CameraFeedIndex road_camera;
  CameraFeedIndex driver_camera;
  CameraFeedIndex wide_road_camera;
  CameraFeedIndex qroad_camera;
  GpsTrace gps_trace;
  std::vector<LogEntry> logs;
  std::vector<TimelineEntry> timeline;
  std::unordered_map<std::string, EnumInfo> enum_info;
  std::unordered_map<std::string, SeriesFormat> series_formats;
  std::string car_fingerprint;
  std::string dbc_name;
  RouteIdentifier route_id;
  bool has_time_range = false;
  double x_min = 0.0;
  double x_max = 1.0;
};

struct StreamExtractBatch {
  std::vector<RouteSeries> series;
  std::vector<CanMessageData> can_messages;
  std::vector<LogEntry> logs;
  std::vector<TimelineEntry> timeline;
  std::unordered_map<std::string, EnumInfo> enum_info;
  std::string car_fingerprint;
  std::string dbc_name;
  bool has_time_offset = false;
  double time_offset = 0.0;
};

struct SketchLayout {
  std::vector<WorkspaceTab> tabs;
  std::vector<std::string> roots;
  int current_tab_index = 0;
};

enum class RouteLoadStage {
  Resolving,
  DownloadingSegment,
  ParsingSegment,
  Finished,
};

struct RouteLoadProgress {
  RouteLoadStage stage = RouteLoadStage::Resolving;
  size_t segment_index = 0;
  size_t segment_count = 0;
  uint64_t current = 0;
  uint64_t total = 0;
  size_t segments_downloaded = 0;
  size_t segments_parsed = 0;
  size_t total_segments = 0;
  uint64_t bytes_downloaded = 0;
  int num_workers = 1;
  std::string segment_name;
};

using RouteLoadProgressCallback = std::function<void(const RouteLoadProgress &)>;

class StreamAccumulator {
public:
  explicit StreamAccumulator(const std::string &dbc_name = {}, std::optional<double> time_offset = std::nullopt);
  ~StreamAccumulator();

  StreamAccumulator(const StreamAccumulator &) = delete;
  StreamAccumulator &operator=(const StreamAccumulator &) = delete;

  void setDbcName(const std::string &dbc_name);
  void appendEvent(cereal::Event::Which which, kj::ArrayPtr<const capnp::word> data);
  void appendCanFrames(CanServiceKind service, const std::vector<LiveCanFrame> &frames);
  StreamExtractBatch takeBatch();
  const std::string &carFingerprint() const;
  const std::string &dbc_name() const;
  std::optional<double> timeOffset() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

SketchLayout load_sketch_layout(const std::filesystem::path &layout_path);
std::vector<std::string> available_dbc_names();
std::vector<std::string> collect_route_roots_for_paths(const std::vector<std::string> &paths);
std::optional<dbc::Database> load_dbc_by_name(const std::string &dbc_name);
std::vector<RouteSeries> decode_can_messages(const std::vector<CanMessageData> &can_messages,
                                             const std::string &dbc_name,
                                             std::unordered_map<std::string, EnumInfo> *enum_info = nullptr);
RouteData load_route_data(const std::string &route_name,
                          const std::string &data_dir = {},
                          const std::string &dbc_name = {},
                          const RouteLoadProgressCallback &progress = {});
RouteIdentifier parse_route_identifier(std::string_view route_name);
void rebuild_gps_trace(RouteData *route_data);

// *****
// icons
// *****

namespace icon {
constexpr const char ARROW_DOWN_UP[]         = "\xef\x84\xa7";
constexpr const char ARROW_LEFT_RIGHT[]      = "\xef\x84\xab";
constexpr const char BAR_CHART[]             = "\xef\x85\xbe";
constexpr const char BOX_ARROW_UP_RIGHT[]    = "\xef\x87\x85";
constexpr const char CLIPBOARD[]             = "\xef\x8a\x90";
constexpr const char CLIPBOARD2[]            = "\xef\x9c\xb3";
constexpr const char DISTRIBUTE_HORIZONTAL[] = "\xef\x8c\x83";
constexpr const char DISTRIBUTE_VERTICAL[]   = "\xef\x8c\x84";
constexpr const char FILE_EARMARK_IMAGE[]    = "\xef\x8d\xad";
constexpr const char FILES[]                 = "\xef\x8f\x82";
constexpr const char INFO_CIRCLE[]           = "\xef\x90\xb1";
constexpr const char PALETTE[]               = "\xef\x92\xb1";
constexpr const char PLUS_SLASH_MINUS[]       = "\xef\x9a\xaa";
constexpr const char SAVE[]                  = "\xef\x94\xa5";
constexpr const char SLIDERS[]               = "\xef\x95\xab";
constexpr const char TRASH[]                 = "\xef\x97\x9e";
constexpr const char X_SQUARE[]              = "\xef\x98\xa9";
constexpr const char ZOOM_OUT[]              = "\xef\x98\xad";
}  // namespace icon

void icon_add_font(float size, bool merge = false);
ImFont *icon_font();
bool icon_menu_item(const char *glyph,
                    const char *label,
                    const char *shortcut = nullptr,
                    bool selected = false,
                    bool enabled = true);

// *****
// app session, UI state, & internal API
// *****

class AsyncRouteLoader;
class CameraFeedView;
class StreamPoller;
class MapDataManager;

enum class SessionDataMode : uint8_t {
  Route,
  Stream,
};

enum class StreamSourceKind : uint8_t {
  CerealLocal,
  CerealRemote,
  Panda,
  SocketCan,
};

struct PandaBusConfig {
  int can_speed_kbps = 500;
  int data_speed_kbps = 2000;
  bool can_fd = false;
};

struct PandaStreamConfig {
  std::string serial;
  std::array<PandaBusConfig, 3> buses = {};
};

struct SocketCanStreamConfig {
  std::string device;
};

struct StreamSourceConfig {
  StreamSourceKind kind = StreamSourceKind::CerealLocal;
  std::string address = "127.0.0.1";
  PandaStreamConfig panda;
  SocketCanStreamConfig socketcan;
};

enum class AppViewMode : uint8_t {
  Plot,
  Cabana,
};

struct BrowserNode {
  std::string label;
  std::string full_path;
  std::vector<BrowserNode> children;
};

struct CabanaSignalSummary {
  std::string path;
  std::string name;
  std::string unit;
  std::string receiver_name;
  std::string comment;
  int start_bit = 0;
  int msb = 0;
  int lsb = 0;
  int size = 0;
  double factor = 1.0;
  double offset = 0.0;
  double min = 0.0;
  double max = 0.0;
  int type = 0;
  int multiplex_value = 0;
  int value_description_count = 0;
  bool is_signed = false;
  bool is_little_endian = false;
  bool has_bit_range = false;
};

struct CabanaMessageSummary {
  std::string root_path;
  std::string service;
  std::string name;
  std::string node;
  std::vector<CabanaSignalSummary> signals;
  int bus = -1;
  uint32_t address = 0;
  int dbc_size = -1;
  bool has_address = false;
  size_t sample_count = 0;
  double frequency_hz = 0.0;
};

struct CabanaSimilarBitMatch {
  std::string message_root;
  std::string label;
  int bus = -1;
  uint32_t address = 0;
  int byte_index = -1;
  int bit_index = -1;
  double score = 0.0;
  double ones_ratio = 0.0;
  double flip_ratio = 0.0;
};

struct CabanaChartState {
  int id = 0;
  int series_type = 0;
  std::vector<std::string> signal_paths;
  std::vector<bool> hidden;
};

struct CabanaChartTabState {
  int id = 0;
  std::vector<CabanaChartState> charts;
};

struct AppSession {
  std::filesystem::path layout_path;
  std::filesystem::path autosave_path;
  std::string route_name;
  std::string data_dir;
  std::string dbc_override;
  StreamSourceConfig stream_source;
  double stream_buffer_seconds = 30.0;
  SessionDataMode data_mode = SessionDataMode::Route;
  RouteIdentifier route_id;
  SketchLayout layout;
  RouteData route_data;
  std::unordered_map<std::string, RouteSeries *> series_by_path;
  std::vector<BrowserNode> browser_nodes;
  std::vector<CabanaMessageSummary> cabana_messages;
  std::unique_ptr<AsyncRouteLoader> route_loader;
  std::unique_ptr<StreamPoller> stream_poller;
  std::array<std::unique_ptr<CameraFeedView>, 4> pane_camera_feeds;
  std::unique_ptr<MapDataManager> map_data;
  bool async_route_loading = false;
  double next_stream_custom_refresh_time = 0.0;
  bool stream_paused = false;
  std::optional<double> stream_time_offset;
};

struct TabUiState {
  struct MapPaneState {
    bool initialized = false;
    bool follow = false;
    float zoom = 1.0f;
    double center_lat = 0.0;
    double center_lon = 0.0;
  };

  struct CameraPaneState {
    bool fit_to_pane = true;
  };

  bool dock_needs_build = true;
  int active_pane_index = 0;
  int runtime_id = 0;
  ImVec2 last_dockspace_size = ImVec2(0.0f, 0.0f);
  std::vector<MapPaneState> map_panes;
  std::vector<CameraPaneState> camera_panes;
};

struct CustomSeriesEditorState {
  bool open = false;
  bool open_help = false;
  bool request_select = false;
  bool selected = false;
  bool focus_name = false;
  int selected_template = 0;
  int selected_additional_source = -1;
  std::string name;
  std::string linked_source;
  std::vector<std::string> additional_sources;
  std::string globals_code;
  std::string function_code = "return value";
  std::string preview_label;
  std::vector<double> preview_xs;
  std::vector<double> preview_ys;
  bool preview_is_result = false;
};

enum class LogTimeMode : uint8_t {
  Route,
  Boot,
  WallClock,
};

struct LogsUiState {
  bool selected = false;
  bool request_select = false;
  bool all_sources = true;
  uint32_t enabled_levels_mask = 0b11110;
  int expanded_index = -1;
  std::string search;
  std::vector<std::string> selected_sources;
  double last_auto_scroll_time = -1.0;
  LogTimeMode time_mode = LogTimeMode::Route;
};

struct CabanaUiState {
  float layout_left_frac = 0.30f;
  float layout_center_frac = 0.32f;
  float layout_center_top_frac = 0.58f;
  float layout_signal_list_frac = 0.56f;
  float layout_right_top_frac = 0.52f;
  bool detail_top_auto_fit = true;
  std::array<char, 128> message_filter = {};
  std::array<char, 32> message_bus_filter = {};
  std::array<char, 48> message_addr_filter = {};
  std::array<char, 64> message_node_filter = {};
  std::array<char, 32> message_freq_filter = {};
  std::array<char, 32> message_count_filter = {};
  std::array<char, 64> message_bytes_filter = {};
  std::array<char, 96> signal_filter = {};
  int sparkline_range_sec = 15;
  bool suppress_defined_signals = false;
  bool sync_message_tabs = true;
  std::string selected_message_root;
  std::string selected_signal_path;
  std::vector<std::string> open_message_roots;
  std::vector<std::string> chart_signal_paths;
  int detail_tab = 0;
  CameraViewKind camera_view = CameraViewKind::Road;
  bool heatmap_live_mode = true;
  bool logs_hex_mode = true;
  int logs_filter_compare = 0;
  std::array<char, 48> logs_filter_value = {};
  bool has_bit_selection = false;
  int selected_bit_byte = -1;
  int selected_bit_index = -1;
  bool binary_drag_active = false;
  bool binary_drag_resizing = false;
  bool binary_drag_moved = false;
  bool binary_drag_signal_is_little_endian = true;
  bool pending_apply_signal_edit = false;
  bool pending_delete_signal = false;
  int binary_drag_press_byte = -1;
  int binary_drag_press_bit = -1;
  int binary_drag_anchor_byte = -1;
  int binary_drag_anchor_bit = -1;
  int binary_drag_current_byte = -1;
  int binary_drag_current_bit = -1;
  std::string binary_drag_signal_path;
  std::string similar_bits_source_root;
  int similar_bits_source_byte = -1;
  int similar_bits_source_bit = -1;
  bool similar_bits_loading = false;
  std::vector<CabanaSimilarBitMatch> similar_bit_matches;
  std::future<std::vector<CabanaSimilarBitMatch>> similar_bit_future;
  std::vector<CabanaChartTabState> chart_tabs;
  std::vector<std::optional<std::pair<double, double>>> chart_zoom_history;
  std::vector<std::optional<std::pair<double, double>>> chart_zoom_redo;
  int next_chart_tab_id = 1;
  int next_chart_id = 1;
  int active_chart_tab = 0;
  int active_chart_index = 0;
  int chart_columns = 1;
  bool chart_scrub_was_playing = false;
  bool chart_zoom_drag_active = false;
  int chart_zoom_drag_chart_id = -1;
  float chart_zoom_drag_plot_min_x = 0.0f;
  float chart_zoom_drag_plot_min_y = 0.0f;
  float chart_zoom_drag_plot_max_x = 0.0f;
  float chart_zoom_drag_plot_max_y = 0.0f;
  float chart_zoom_drag_start_x = 0.0f;
  bool chart_timeline_zoom_drag_active = false;
  float chart_timeline_zoom_start_x = 0.0f;
  float chart_timeline_zoom_min_x = 0.0f;
  float chart_timeline_zoom_max_x = 0.0f;
  double chart_timeline_zoom_range_min = 0.0;
  double chart_timeline_zoom_range_max = 0.0;
  double chart_hover_sec = -1.0;
};

struct AxisLimitsEditorState {
  bool open = false;
  int pane_index = -1;
  double x_min = 0.0;
  double x_max = 1.0;
  bool y_min_enabled = false;
  bool y_max_enabled = false;
  double y_min = 0.0;
  double y_max = 1.0;
};

struct DbcEditorState {
  bool open = false;
  bool loaded = false;
  std::string source_name;
  std::string source_path;
  std::string save_name;
  std::string text;
};

struct CabanaSignalEditorState {
  bool open = false;
  bool loaded = false;
  bool creating = false;
  std::string message_root;
  std::string message_name;
  std::string service;
  std::string signal_path;
  int bus = -1;
  uint32_t message_address = 0;
  std::string original_signal_name;
  std::string signal_name;
  int start_bit = 0;
  int size = 1;
  double factor = 1.0;
  double offset = 0.0;
  double min = 0.0;
  double max = 0.0;
  bool is_signed = false;
  bool is_little_endian = true;
  int type = 0;
  int multiplex_value = 0;
  std::string receiver_name;
  std::string unit;
};

enum class TimelineDragMode : uint8_t {
  None,
  ScrubCursor,
  PanViewport,
  ResizeLeft,
  ResizeRight,
};

struct UndoStack {
  static constexpr size_t kMaxHistory = 50;

  std::vector<SketchLayout> history;
  int position = -1;

  void reset(const SketchLayout &layout) {
    history.clear();
    history.push_back(layout);
    position = 0;
  }

  void push(const SketchLayout &layout) {
    if (position < 0) {
      reset(layout);
      return;
    }
    if (position + 1 < static_cast<int>(history.size())) {
      history.resize(static_cast<size_t>(position + 1));
    }
    history.push_back(layout);
    if (history.size() > kMaxHistory) {
      history.erase(history.begin());
    }
    position = static_cast<int>(history.size()) - 1;
  }

  bool can_undo() const {
    return position > 0;
  }

  bool can_redo() const {
    return position >= 0 && position + 1 < static_cast<int>(history.size());
  }

  const SketchLayout &undo() {
    return history[static_cast<size_t>(--position)];
  }

  const SketchLayout &redo() {
    return history[static_cast<size_t>(++position)];
  }
};

struct UiState {
  bool open_open_route = false;
  bool open_stream = false;
  bool open_load_layout = false;
  bool open_save_layout = false;
  bool open_preferences = false;
  bool open_find_signal = false;
  bool request_close = false;
  bool request_reset_layout = false;
  bool request_save_layout = false;
  bool request_new_tab = false;
  bool request_duplicate_tab = false;
  bool request_close_tab = false;
  bool follow_latest = false;
  bool cabana_mode_initialized = false;
  bool has_shared_range = false;
  bool has_tracker_time = false;
  bool layout_dirty = false;
  bool playback_loop = false;
  bool playback_playing = false;
  bool show_deprecated_fields = false;
  bool show_fps_overlay = false;
  bool fps_overlay_initialized = false;
  bool view_mode_initialized = false;
  bool start_cabana = false;
  bool suppress_range_side_effects = false;
  bool browser_nodes_dirty = false;
  int active_tab_index = 0;
  int next_tab_runtime_id = 1;
  int requested_tab_index = -1;
  int rename_tab_index = -1;
  AppViewMode view_mode = AppViewMode::Plot;
  bool focus_rename_tab_input = false;
  std::vector<TabUiState> tabs;
  std::array<char, 128> route_buffer = {};
  std::array<char, 128> stream_address_buffer = {};
  std::array<char, 128> panda_serial_buffer = {};
  std::array<char, 128> socketcan_device_buffer = {};
  std::array<char, 128> rename_tab_buffer = {};
  std::array<char, 128> browser_filter = {};
  std::array<char, 512> data_dir_buffer = {};
  std::array<char, 512> load_layout_buffer = {};
  std::array<char, 512> save_layout_buffer = {};
  std::array<char, 256> find_signal_buffer = {};
  std::string selected_browser_path;
  std::vector<std::string> selected_browser_paths;
  std::string browser_selection_anchor;
  std::string route_slice_buffer;
  std::string error_text;
  bool open_error_popup = false;
  std::string status_text = "Ready";
  std::string route_copy_feedback_text;
  double route_copy_feedback_until = 0.0;
  bool editing_route_slice = false;
  bool focus_route_slice_input = false;
  StreamSourceKind stream_source_kind = StreamSourceKind::CerealLocal;
  std::array<int, 3> panda_can_speed_kbps = {500, 500, 500};
  std::array<int, 3> panda_data_speed_kbps = {2000, 2000, 2000};
  std::array<bool, 3> panda_can_fd = {false, false, false};
  float sidebar_width = 320.0f;
  double route_x_min = 0.0;
  double route_x_max = 1.0;
  double x_view_min = 0.0;
  double x_view_max = 1.0;
  double tracker_time = 0.0;
  double playback_rate = 1.0;
  double playback_step = 0.1;
  double stream_buffer_seconds = 30.0;
  TimelineDragMode timeline_drag_mode = TimelineDragMode::None;
  double timeline_drag_anchor_time = 0.0;
  double timeline_drag_anchor_x_min = 0.0;
  double timeline_drag_anchor_x_max = 0.0;
  AxisLimitsEditorState axis_limits;
  DbcEditorState dbc_editor;
  CabanaSignalEditorState cabana_signal_editor;
  CabanaUiState cabana;
  CustomSeriesEditorState custom_series;
  LogsUiState logs;
  UndoStack undo;
};

// inline helpers

inline ImVec4 color_rgb(int r, int g, int b, float alpha = 1.0f) {
  return ImVec4(static_cast<float>(r) / 255.0f,
                static_cast<float>(g) / 255.0f,
                static_cast<float>(b) / 255.0f,
                alpha);
}

inline ImVec4 color_rgb(const std::array<uint8_t, 3> &color, float alpha = 1.0f) {
  return color_rgb(color[0], color[1], color[2], alpha);
}

inline std::string trim_copy(std::string_view text) {
  size_t begin = 0;
  size_t end = text.size();
  while (begin < end && std::isspace(static_cast<unsigned char>(text[begin]))) {
    ++begin;
  }
  while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1]))) {
    --end;
  }
  return std::string(text.substr(begin, end - begin));
}

inline std::string lowercase(std::string_view value) {
  std::string out(value);
  std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return out;
}

inline int imgui_resize_callback(ImGuiInputTextCallbackData *data) {
  if (data->EventFlag != ImGuiInputTextFlags_CallbackResize || data->UserData == nullptr) return 0;
  auto *text = static_cast<std::string *>(data->UserData);
  text->resize(static_cast<size_t>(data->BufTextLen));
  data->Buf = text->data();
  return 0;
}

inline bool input_text_string(const char *label,
                              std::string *text,
                              ImGuiInputTextFlags flags = 0) {
  flags |= ImGuiInputTextFlags_CallbackResize;
  return ImGui::InputText(label, text->data(), text->capacity() + 1,
                          flags, imgui_resize_callback, text);
}

inline bool input_text_with_hint_string(const char *label,
                                        const char *hint,
                                        std::string *text,
                                        ImGuiInputTextFlags flags = 0) {
  flags |= ImGuiInputTextFlags_CallbackResize;
  return ImGui::InputTextWithHint(label, hint, text->data(), text->capacity() + 1,
                                  flags, imgui_resize_callback, text);
}

inline bool input_text_multiline_string(const char *label,
                                        std::string *text,
                                        const ImVec2 &size = ImVec2(0.0f, 0.0f),
                                        ImGuiInputTextFlags flags = 0) {
  flags |= ImGuiInputTextFlags_CallbackResize;
  return ImGui::InputTextMultiline(label, text->data(), text->capacity() + 1,
                                   size, flags, imgui_resize_callback, text);
}

inline bool is_local_stream_address(std::string_view address) {
  return address.empty() || address == "127.0.0.1" || address == "localhost";
}

inline void ensure_parent_dir(const std::filesystem::path &path) {
  if (path.has_parent_path()) {
    std::filesystem::create_directories(path.parent_path());
  }
}

inline std::string shell_quote(std::string_view value) {
  std::string quoted;
  quoted.reserve(value.size() + 8);
  quoted.push_back('\'');
  for (const char c : value) {
    if (c == '\'') {
      quoted += "'\\''";
    } else {
      quoted.push_back(c);
    }
  }
  quoted.push_back('\'');
  return quoted;
}

// app.cc public API

const WorkspaceTab *app_active_tab(const SketchLayout &layout, const UiState &state);
WorkspaceTab *app_active_tab(SketchLayout *layout, const UiState &state);
TabUiState *app_active_tab_state(UiState *state);

void app_push_mono_font();
void app_pop_mono_font();
bool app_add_curve_to_active_pane(AppSession *session, UiState *state, const std::string &path);

std::string app_curve_display_name(const Curve &curve);
std::array<uint8_t, 3> app_next_curve_color(const Pane &pane);
const RouteSeries *app_find_route_series(const AppSession &session, const std::string &path);
void app_decimate_samples(const std::vector<double> &xs_in,
                          const std::vector<double> &ys_in,
                          int max_points,
                          std::vector<double> *xs_out,
                          std::vector<double> *ys_out);
std::optional<double> app_sample_xy_value_at_time(const std::vector<double> &xs,
                                                   const std::vector<double> &ys,
                                                   bool stairs,
                                                   double tm);
void save_layout_json(const SketchLayout &layout, const std::filesystem::path &path);

// *****
// browser
// *****

void rebuild_route_index(AppSession *session);
void rebuild_browser_nodes(AppSession *session, UiState *state);
SeriesFormat compute_series_format(const std::vector<double> &values, bool enum_like = false);
std::string format_display_value(double display_value,
                                 const SeriesFormat &format,
                                 const EnumInfo *enum_info);
std::vector<std::string> decode_browser_drag_payload(std::string_view payload);
void collect_visible_leaf_paths(const BrowserNode &node,
                                const std::string &filter,
                                std::vector<std::string> *out);
void draw_browser_node(AppSession *session,
                       const BrowserNode &node,
                       UiState *state,
                       const std::string &filter,
                       const std::vector<std::string> &visible_paths);

// *****
// custom series
// *****

void open_custom_series_editor(UiState *state, const std::string &preferred_source = {});
std::string preferred_custom_series_source(const Pane &pane);
void refresh_all_custom_curves(AppSession *session, UiState *state);
void draw_custom_series_editor(AppSession *session, UiState *state);

// *****
// logs
// *****

void draw_logs_tab(AppSession *session, UiState *state);

// *****
// map
// *****

void draw_map_pane(AppSession *session, UiState *state, Pane *pane, int pane_index);

// *****
// runtime (GLFW, async loaders, streaming, camera)
// *****

struct GLFWwindow;

struct RouteLoadSnapshot {
  bool active = false;
  size_t total_segments = 0;
  size_t segments_downloaded = 0;
  size_t segments_parsed = 0;
};

struct StreamPollSnapshot {
  bool active = false;
  bool connected = false;
  bool paused = false;
  StreamSourceKind source_kind = StreamSourceKind::CerealLocal;
  std::string source_label;
  std::string dbc_name;
  std::string car_fingerprint;
  double buffer_seconds = 30.0;
  uint64_t received_messages = 0;
};

class GlfwRuntime {
public:
  explicit GlfwRuntime(const Options &options);
  ~GlfwRuntime();

  GlfwRuntime(const GlfwRuntime &) = delete;
  GlfwRuntime &operator=(const GlfwRuntime &) = delete;

  GLFWwindow *window() const;

private:
  GLFWwindow *window_ = nullptr;
};

class ImGuiRuntime {
public:
  explicit ImGuiRuntime(GLFWwindow *window);
  ~ImGuiRuntime();

  ImGuiRuntime(const ImGuiRuntime &) = delete;
  ImGuiRuntime &operator=(const ImGuiRuntime &) = delete;
};

class TerminalRouteProgress {
public:
  explicit TerminalRouteProgress(bool enabled);
  ~TerminalRouteProgress();

  TerminalRouteProgress(const TerminalRouteProgress &) = delete;
  TerminalRouteProgress &operator=(const TerminalRouteProgress &) = delete;

  void update(const RouteLoadProgress &progress);
  void finish();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class AsyncRouteLoader {
public:
  explicit AsyncRouteLoader(bool enable_terminal_progress);
  ~AsyncRouteLoader();

  AsyncRouteLoader(const AsyncRouteLoader &) = delete;
  AsyncRouteLoader &operator=(const AsyncRouteLoader &) = delete;

  void start(const std::string &route_name, const std::string &data_dir, const std::string &dbc_name);
  RouteLoadSnapshot snapshot() const;
  bool consume(RouteData *route_data, std::string *error_text);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class StreamPoller {
public:
  StreamPoller();
  ~StreamPoller();

  StreamPoller(const StreamPoller &) = delete;
  StreamPoller &operator=(const StreamPoller &) = delete;

  void start(const StreamSourceConfig &source,
             double buffer_seconds,
             const std::string &dbc_name,
             std::optional<double> time_offset = std::nullopt);
  void setPaused(bool paused);
  void stop();
  StreamPollSnapshot snapshot() const;
  bool consume(StreamExtractBatch *batch, std::string *error_text);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class CameraFeedView {
public:
  CameraFeedView();
  ~CameraFeedView();

  CameraFeedView(const CameraFeedView &) = delete;
  CameraFeedView &operator=(const CameraFeedView &) = delete;

  void setRouteData(const RouteData &route_data);
  void setCameraIndex(const CameraFeedIndex &camera_index, CameraViewKind view);
  void update(double tracker_time);
  void draw(float width, bool loading);
  void drawSized(ImVec2 size, bool loading, bool fit_to_pane = false);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
