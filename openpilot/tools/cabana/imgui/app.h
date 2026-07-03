#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <string>

#include "imgui.h"

#include "tools/cabana/streams/abstractstream.h"

struct GLFWwindow;

struct Options {
  // stream selection (parity with tools/cabana/cabana.cc)
  std::string route;
  std::string data_dir;
  std::string dbc_path;
  std::string panda_serial;
  std::string socketcan_device;
  std::string zmq_address;
  bool msgq = false;
  bool panda = false;
  bool auto_source = false;
  bool qcam = false;
  bool ecam = false;
  bool dcam = false;
  bool no_vipc = false;

  // window / capture
  int width = 1600;
  int height = 900;
  bool show = true;
  std::optional<bool> dark_theme;  // set only when --theme is passed on the CLI
  std::string output_path;
};

enum class Theme {
  Light,
  Dark,
};

class GlfwRuntime {
public:
  explicit GlfwRuntime(const Options &options);
  ~GlfwRuntime();
  GLFWwindow *window() const { return window_; }

private:
  GLFWwindow *window_ = nullptr;
};

class ImGuiRuntime {
public:
  explicit ImGuiRuntime(GLFWwindow *window);
  ~ImGuiRuntime();
};

void save_framebuffer_png(const std::filesystem::path &output_path, int width, int height);

// Shared UI state for the imgui shell, handed to every panel by reference.
// Panels are free functions `void draw_xxx_panel(AppState &app)`, each owning
// its own ImGui::Begin/End, declared below and defined in their own .cc file.
struct AppState {
  std::unique_ptr<AbstractStream> stream;
  std::optional<MessageId> selected_msg_id;  // set by the messages panel; consumed by the detail panel
  std::vector<MessageId> open_msg_tabs;      // detail-view tab strip, most recent selection last
  Theme theme = Theme::Light;
  bool theme_changed = false;
  bool show_about = false;
  bool request_close = false;
};

inline bool has_stream(const AppState &app) {
  return dynamic_cast<DummyStream *>(app.stream.get()) == nullptr;
}

constexpr float TRANSPORT_BAR_HEIGHT = 40.0f;

// dock window titles -- mirror tools/cabana/mainwin.cc dock titles exactly
constexpr const char *MESSAGES_WINDOW_TITLE = "MESSAGES";
constexpr const char *VIDEO_WINDOW_TITLE = "Video";
constexpr const char *CHARTS_WINDOW_TITLE = "Charts";
constexpr const char *CENTER_WINDOW_TITLE = "Detail";

// bootstrap-icons glyphs merged into the UI fonts by theme.cc load_fonts()
namespace icon {
constexpr const char PLAY_FILL[] = "\xef\x93\xb4";
constexpr const char PAUSE_FILL[] = "\xef\x93\x83";
}  // namespace icon

// panels (each in its own imgui/*.cc file)
void draw_messages_panel(AppState &app);
void draw_detail_panel(AppState &app);  // center: message tabs + binary view + logs; welcome when no selection
void draw_history_log(AppState &app);   // "Logs" tab content, called from the detail panel
void draw_video_panel(AppState &app);
void draw_charts_panel(AppState &app);
void draw_transport_bar(AppState &app);

// tool windows/dialogs: each owns its file; open_*() requests it, draw_*()
// runs every frame from app.cc
void draw_stream_selector(AppState &app);  // stream_selector.cc
void open_stream_selector();
void draw_find_signal(AppState &app);  // find_signal.cc
void open_find_signal();
void draw_find_similar_bits(AppState &app);  // find_similar_bits.cc
void open_find_similar_bits();
void draw_settings_dialog(AppState &app);  // settings_dialog.cc
void open_settings_dialog();
void draw_help_overlay(AppState &app);  // help_overlay.cc
void toggle_help_overlay();
void draw_route_tools(AppState &app);  // routes_dialog.cc: remote route browser + CSV export
void open_remote_route_browser(std::string *out_route);  // picked route written to *out_route on confirm
void open_export_csv();

// charts API (defined in charts_panel.cc; panel state is file-local there)
void charts_show_signal(const MessageId &id, const cabana::Signal *sig, bool show);
bool charts_is_showing(const MessageId &id, const cabana::Signal *sig);

// shared widgets (defined in messages_panel.cc)
ImU32 to_im_color(const ColorRGBA &c);
// hex byte cells as in Qt's MessageBytesDelegate; colors may be null (no highlight)
void draw_message_bytes(const uint8_t *dat, size_t size, const ColorRGBA *colors, bool multiple_lines, float avail_width);

// theme.cc
const std::filesystem::path &repo_root();
void load_fonts();
void apply_theme(Theme theme);
Theme theme_from_settings();
ImVec4 theme_clear_color();
void push_bold_font(float size = 0.0f);
void pop_bold_font();
void push_mono_font(float size = 0.0f);
void pop_mono_font();

// app.cc
int run(const Options &options, std::unique_ptr<AbstractStream> stream);
