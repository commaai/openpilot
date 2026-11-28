#pragma once

#include "tools/jotpluggler/common.h"
#include "tools/jotpluggler/map.h"

#include <filesystem>
#include <functional>
#include <optional>

struct GLFWwindow;

enum class PaneDropZone {
  Center,
  Left,
  Right,
  Top,
  Bottom,
};

enum class PaneMenuActionKind {
  None,
  OpenAxisLimits,
  OpenCustomSeries,
  SplitLeft,
  SplitRight,
  SplitTop,
  SplitBottom,
  ResetView,
  Clear,
  Close,
};

struct PaneMenuAction {
  PaneMenuActionKind kind = PaneMenuActionKind::None;
  int pane_index = -1;
};

struct PaneCurveDragPayload {
  int tab_index = -1;
  int pane_index = -1;
  int curve_index = -1;
};

struct PaneDropAction {
  PaneDropZone zone = PaneDropZone::Center;
  int target_pane_index = -1;
  bool from_browser = false;
  std::vector<std::string> browser_paths;
  std::string special_item_id;
  PaneCurveDragPayload curve_ref;
};

inline constexpr float SIDEBAR_WIDTH = 320.0f;
inline constexpr float SIDEBAR_MIN_WIDTH = 220.0f;
inline constexpr float SIDEBAR_MAX_WIDTH = 520.0f;
inline constexpr float TIMELINE_BAR_HEIGHT = 14.0f;
inline constexpr float STATUS_BAR_HEIGHT = 52.0f;
inline constexpr double MIN_HORIZONTAL_ZOOM_SECONDS = 2.0;
inline constexpr double PLOT_Y_PADDING_FRACTION = 0.05;

struct UiMetrics {
  float width = 0.0f;
  float height = 0.0f;
  float top_offset = 0.0f;
  float sidebar_width = SIDEBAR_WIDTH;
  float content_x = 0.0f;
  float content_y = 0.0f;
  float content_w = 0.0f;
  float content_h = 0.0f;
  float status_bar_y = 0.0f;
};

std::filesystem::path resolve_layout_path(const std::string &layout_arg);
std::filesystem::path autosave_path_for_layout(const std::filesystem::path &layout_path);
std::vector<std::string> available_layout_names();

SketchLayout make_empty_layout();
void cancel_rename_tab(UiState *state);
void sync_ui_state(UiState *state, const SketchLayout &layout);
void sync_route_buffers(UiState *state, const AppSession &session);
void sync_stream_buffers(UiState *state, const AppSession &session);
void sync_layout_buffers(UiState *state, const AppSession &session);
void mark_all_docks_dirty(UiState *state);
void clear_layout_autosave(const AppSession &session);
bool autosave_layout(AppSession *session, UiState *state);
bool apply_axis_limits_editor(AppSession *session, UiState *state);
void open_axis_limits_editor(const AppSession &session, UiState *state, int pane_index);
void persist_shared_range_to_tab(WorkspaceTab *tab, const UiState &state);
void clear_pane_vertical_limits(Pane *pane);

void refresh_replaced_layout_ui(AppSession *session, UiState *state, bool mark_docks);
void start_new_layout(AppSession *session, UiState *state, const std::string &status_text = "New untitled layout");
void apply_dbc_override_change(AppSession *session, UiState *state, const std::string &dbc_override);

void app_push_bold_font();
void app_pop_bold_font();
void draw_vertical_splitter(const char *id, float height, float min_left, float max_left, float *left_width);
void draw_right_splitter(const char *id, float height, float min_right, float max_right, float *right_width);
bool draw_horizontal_splitter(const char *id, float width, float min_top, float max_top, float *top_height);
void draw_payload_bytes(std::string_view data, const std::string *prev_data = nullptr);
void draw_payload_preview_boxes(const char *id, std::string_view data, const std::string *prev_data, float max_width);
void draw_signal_sparkline(const AppSession &session,
                           const UiState &state,
                           std::string_view signal_path,
                           bool selected,
                           ImVec2 size = ImVec2(0.0f, 24.0f));
ImU32 mix_color(ImU32 a, ImU32 b, float t);
void draw_empty_panel(const char *title, const char *message);

UiMetrics compute_ui_metrics(const ImVec2 &size, float top_offset, float sidebar_width);
void draw_sidebar(AppSession *session, const UiMetrics &ui, UiState *state, bool show_camera_feed);
void draw_workspace(AppSession *session, const UiMetrics &ui, UiState *state);
void draw_pane_windows(AppSession *session, UiState *state);

// plot.cc
void draw_plot(const AppSession &session, Pane *pane, UiState *state);
bool draw_pane_close_button_overlay();
void draw_pane_frame_overlay();
std::optional<PaneMenuAction> draw_pane_context_menu(const WorkspaceTab &tab, int pane_index);
bool curve_has_samples(const AppSession &session, const Curve &curve);
bool curve_has_local_samples(const Curve &curve);
std::string app_curve_display_name(const Curve &curve);
bool mark_layout_dirty(AppSession *session, UiState *state);

const RouteSeries *app_find_route_series(const AppSession &session, const std::string &path);
void sync_camera_feeds(AppSession *session);
void apply_route_data(AppSession *session, UiState *state, RouteData route_data);
bool apply_undo(AppSession *session, UiState *state);
bool apply_redo(AppSession *session, UiState *state);
bool infer_stream_follow_state(const UiState &state, const AppSession &session);
void ensure_shared_range(UiState *state, const AppSession &session);
void clamp_shared_range(UiState *state, const AppSession &session);
void reset_shared_range(UiState *state, const AppSession &session);
void update_follow_range(UiState *state, const AppSession &session);
void advance_playback(UiState *state, const AppSession &session);
void step_tracker(UiState *state, double direction);
std::string dbc_combo_label(const AppSession &session);
const char *log_selector_name(LogSelector selector);
const char *log_selector_description(LogSelector selector);
std::string format_cache_bytes(uint64_t bytes);
MapCacheStats directory_cache_stats(const std::filesystem::path &root);
float draw_main_menu_bar(AppSession *session, UiState *state);

bool reset_layout(AppSession *session, UiState *state);
bool reload_layout(AppSession *session, UiState *state, const std::string &layout_arg);
bool save_layout(AppSession *session, UiState *state, const std::string &layout_path);
void rebuild_session_route_data(AppSession *session, UiState *state,
                                const RouteLoadProgressCallback &progress = {});
void stop_stream_session(AppSession *session, UiState *state, bool preserve_data = true);
bool start_stream_session(AppSession *session,
                          UiState *state,
                          const StreamSourceConfig &source,
                          double buffer_seconds,
                          bool preserve_existing_data = false);
void start_async_route_load(AppSession *session, UiState *state);
void poll_async_route_load(AppSession *session, UiState *state);
bool reload_session(AppSession *session, UiState *state, const std::string &route_name, const std::string &data_dir);
void draw_popups(AppSession *session, UiState *state);

void draw_status_bar(const AppSession &session, const UiMetrics &ui, UiState *state);
void draw_sidebar_resizer(const UiMetrics &ui, UiState *state);

void apply_stream_batch(AppSession *session, UiState *state, StreamExtractBatch batch);

void render_frame(GLFWwindow *window, AppSession *session, UiState *state, const std::filesystem::path *capture_path);
