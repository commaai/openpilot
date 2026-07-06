#include "tools/loggy/shell/runtime.h"

#include "tools/loggy/backend/session.h"
#include "tools/loggy/panes/map.h"
#include "tools/loggy/shell/live_source_controls.h"
#include "tools/loggy/shell/native_dialog.h"
#include "tools/loggy/shell/settings_ui.h"
#include "tools/loggy/shell/remote_routes.h"
#include "tools/loggy/shell/route_controls.h"
#include "tools/loggy/shell/theme.h"
#include "tools/loggy/shell/workspace.h"

#include "json11/json11.hpp"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_opengl3_loader.h"
#include "implot.h"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>


namespace loggy {
namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using SignalHandler = void (*)(int);

volatile std::sig_atomic_t g_shutdown_signal = 0;

void shutdown_signal_handler(int signal) {
  g_shutdown_signal = signal;
}

bool shutdown_requested() {
  return g_shutdown_signal != 0;
}

class ShutdownSignalHandlers {
public:
  ShutdownSignalHandlers() {
    g_shutdown_signal = 0;
    previous_sigint_ = std::signal(SIGINT, shutdown_signal_handler);
    previous_sigterm_ = std::signal(SIGTERM, shutdown_signal_handler);
  }

  ~ShutdownSignalHandlers() {
    if (previous_sigint_ != SIG_ERR) std::signal(SIGINT, previous_sigint_);
    if (previous_sigterm_ != SIG_ERR) std::signal(SIGTERM, previous_sigterm_);
  }

private:
  SignalHandler previous_sigint_ = SIG_DFL;
  SignalHandler previous_sigterm_ = SIG_DFL;
};

bool g_escape_pressed = false;

bool modal_escape_pressed() {
  return g_escape_pressed;
}

void loggy_key_callback(GLFWwindow *window, int key, int scancode, int action, int mods) {
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
    g_escape_pressed = true;
  }
  ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);
}

void glfw_error_callback(int error, const char *description) {
  const std::string_view desc = description != nullptr ? description : "unknown";
  if (error == 65539 && desc.find("Invalid window attribute 0x0002000D") != std::string_view::npos) {
    return;
  }
  std::cerr << "GLFW error " << error << ": " << desc << "\n";
}

class GlfwRuntime {
public:
  explicit GlfwRuntime(const Options &options) {
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) throw std::runtime_error("glfwInit failed");

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
#endif
    const bool fixed_size = !options.show;
    glfwWindowHint(GLFW_RESIZABLE, fixed_size ? GLFW_FALSE : GLFW_TRUE);
    glfwWindowHint(GLFW_VISIBLE, options.show ? GLFW_TRUE : GLFW_FALSE);

    window_ = glfwCreateWindow(options.width, options.height, "loggy", nullptr, nullptr);
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

  ~GlfwRuntime() {
    if (window_ != nullptr) {
      glfwDestroyWindow(window_);
    }
    glfwTerminate();
  }

  GLFWwindow *window() const { return window_; }

private:
  GLFWwindow *window_ = nullptr;
};

class ImGuiRuntime {
public:
  explicit ImGuiRuntime(GLFWwindow *window) {
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
    glfwSetKeyCallback(window, loggy_key_callback);
    if (!ImGui_ImplOpenGL3_Init("#version 330")) {
      ImGui_ImplGlfw_Shutdown();
      ImPlot::DestroyContext();
      ImGui::DestroyContext();
      throw std::runtime_error("ImGui_ImplOpenGL3_Init failed");
    }

    load_fonts();
    apply_theme();
  }

  ~ImGuiRuntime() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
  }
};

struct FrameStats {
  // Fixed-size ring buffer holding the last kWindow seconds (capped at kCapacity
  // samples so slow frames can't grow it): no per-frame heap allocation, and aging
  // by wall time (not sample count) keeps the window correct at any actual fps.
  static constexpr std::chrono::duration<double> kWindow = std::chrono::duration<double>(5.0);
  static constexpr size_t kCapacity = 2048;
  struct Sample {
    Clock::time_point at;
    double ms;
  };
  std::array<Sample, kCapacity> samples{};
  size_t head = 0;
  size_t count = 0;
  double latest_ms = 0.0;

  void add(double ms) {
    const Clock::time_point now = Clock::now();
    latest_ms = ms;
    samples[head] = {now, ms};
    head = (head + 1) % kCapacity;
    count = std::min(count + 1, kCapacity);
    while (count > 0 && now - samples[(head + kCapacity - count) % kCapacity].at > kWindow) {
      --count;
    }
  }

  void reset() {
    count = 0;
    head = 0;
    latest_ms = 0.0;
  }

  double p99_ms() const {
    if (count == 0) return 0.0;
    std::array<double, kCapacity> values;
    for (size_t i = 0; i < count; ++i) {
      values[i] = samples[(head + kCapacity - count + i) % kCapacity].ms;
    }
    const size_t idx = std::min(count - 1, static_cast<size_t>(count * 0.99));
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(idx),
                     values.begin() + static_cast<std::ptrdiff_t>(count));
    return values[idx];
  }
};

struct AppState {
  explicit AppState(const Options &opts)
    : options(opts),
      session(SessionConfig{
        .preset = opts.preset,
        .layout = opts.layout,
        .route_name = opts.route_name,
        .data_dir = opts.data_dir,
        .settings_path = opts.settings_path,
        .stream_address = opts.stream_address,
        .stream_source_kind = opts.stream_source_kind,
        .stream_panda_buses = opts.stream_panda_buses,
        .stream_buffer_seconds = opts.stream_buffer_seconds,
        .stream = opts.stream,
      }),
      show_frame_hud(opts.show_frame_hud && session.settings.show_frame_hud),
      target_fps(std::clamp(session.settings.target_fps, kMinLoggyTargetFps, kMaxLoggyTargetFps)),
      theme_kind(loggy_theme_from_name(session.settings.theme)) {
    apply_theme(theme_kind);
    sync_live_source_fields(session, live_source_ui);
    sync_route_popup_fields(session, route_ui);
    workspace_history.reset(session.workspace);
    workspace_select_request = session.workspace.current_tab_index;
    if (session.loaded_workspace_draft) {
      workspace_status = "Loaded workspace draft";
    }
  }

  const Options &options;
  Session session;
  FrameStats frame_stats;
  Clock::time_point last_playback_update = Clock::now();
  bool show_frame_hud = true;
  int target_fps = kDefaultLoggyTargetFps;
  LoggyThemeKind theme_kind = LoggyThemeKind::Darcula;
  bool show_demo = false;
  bool request_close = false;
  WorkspaceHistory workspace_history;
  std::string workspace_status;
  int workspace_select_request = -1;
  bool open_help_popup = false;
  LiveSourceUiState live_source_ui;
  RouteUiState route_ui;
  SettingsUiState settings_ui;

  struct PaneAction {
    enum class Type { Split, Close };
    Type type = Type::Split;
    int tab_index = -1;
    int pane_index = -1;
    PaneSplit split = PaneSplit::Right;
  };
  std::optional<PaneAction> pending_pane_action;
};

void request_help_popup(AppState &app) {
  app.open_help_popup = true;
}

void draw_help_popup(AppState &app) {
  if (app.open_help_popup) {
    ImGui::OpenPopup("Help");
    app.open_help_popup = false;
  }

  if (!ImGui::BeginPopupModal("Help", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) return;
  const bool close_requested = modal_escape_pressed();

  ImGui::TextUnformatted("Loggy");
  ImGui::TextDisabled("Route analysis shell for CAN + qlog and live streams.");
  ImGui::Separator();

  ImGui::TextUnformatted("Shortcuts");
  ImGui::BulletText("F1: Open this Help/About overlay.");
  ImGui::BulletText("Space: Play/Pause playback.");
  ImGui::BulletText("F12: Toggle frame-time HUD.");
  ImGui::BulletText("Mouse: Drag timeline to seek.");
  ImGui::BulletText("Route mode: use the footer Route button to reopen a route.");

  ImGui::Spacing();
  ImGui::TextUnformatted("Routes");
  ImGui::BulletText("Route popup: File → Open Route or Status footer Route button.");
  ImGui::BulletText("Route format: <dongle>|<timestamp>[:<begin>[:<end>]] with optional auto/rlog/qlog selector.");
  ImGui::BulletText("Live mode: File → Live Source or Route footer Open Live button.");
  ImGui::BulletText("Live controls: Source, Pause Live, and Follow live in the status bar.");
  ImGui::BulletText("Slice quickly with `N`, `N:`, or `N:M` (route popup).");

  ImGui::Spacing();
  ImGui::TextUnformatted("Workspace");
  ImGui::BulletText("Preset `cabana` starts with Messages/Binary/Signal analysis panes.");
  ImGui::BulletText("Preset `jotpluggler` starts with Browser/Plot/Logs workspace.");
  ImGui::BulletText("Context-click pane for split and close actions; use `+` to add workspace tabs.");
  ImGui::BulletText("Use File → Open Route / Live Source and route popup actions to change data sources.");
  ImGui::Spacing();
  if (ImGui::Button("Close") || close_requested) ImGui::CloseCurrentPopup();
  ImGui::EndPopup();
}

void save_framebuffer_png(const fs::path &output_path, int width, int height) {
  if (width <= 0 || height <= 0) throw std::runtime_error("Invalid framebuffer size");
  if (output_path.has_parent_path()) {
    fs::create_directories(output_path.parent_path());
  }

  std::vector<uint8_t> pixels(static_cast<size_t>(width) * static_cast<size_t>(height) * 4U, 0);
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());

  const fs::path ppm_path = output_path.parent_path() / (output_path.stem().string() + ".ppm");
  char header[64];
  const int header_len = std::snprintf(header, sizeof(header), "P6\n%d %d\n255\n", width, height);
  std::string ppm(header, static_cast<size_t>(header_len));
  ppm.reserve(ppm.size() + static_cast<size_t>(width) * static_cast<size_t>(height) * 3U);
  for (int y = height - 1; y >= 0; --y) {
    for (int x = 0; x < width; ++x) {
      const size_t index = static_cast<size_t>((y * width + x) * 4);
      ppm.append(reinterpret_cast<const char *>(&pixels[index]), 3);
    }
  }

  std::ofstream out(ppm_path, std::ios::binary);
  out.write(ppm.data(), static_cast<std::streamsize>(ppm.size()));
  if (!out.good()) throw std::runtime_error("Failed to write " + ppm_path.string());
  out.close();

  const std::string command = "convert " + shell_quote(ppm_path.string()) + " " + shell_quote(output_path.string());
  if (std::system(command.c_str()) != 0) throw std::runtime_error("image conversion failed: " + command);
  fs::remove(ppm_path);
}

void draw_frame_hud(const FrameStats &stats) {
  const ImGuiViewport *viewport = ImGui::GetMainViewport();
  const std::string label = "CPU " + std::to_string(stats.latest_ms).substr(0, 4) + " ms   p99 "
                          + std::to_string(stats.p99_ms()).substr(0, 4) + " ms";
  const ImVec2 padding(10.0f, 8.0f);
  const ImVec2 margin(12.0f, 10.0f);

  push_mono_font();
  ImFont *font = ImGui::GetFont();
  const float font_size = ImGui::GetFontSize();
  const ImVec2 text_size = ImGui::CalcTextSize(label.c_str());
  pop_mono_font();

  const ImVec2 size(text_size.x + padding.x * 2.0f, text_size.y + padding.y * 2.0f);
  const ImVec2 pos(viewport->Pos.x + viewport->Size.x - size.x - margin.x, viewport->Pos.y + margin.y);
  const ImVec2 max(pos.x + size.x, pos.y + size.y);
  ImDrawList *draw_list = ImGui::GetForegroundDrawList();
  draw_list->AddRectFilled(pos, max, ImGui::GetColorU32(color_rgb(43, 43, 43, 0.92f)), 4.0f);
  draw_list->AddRect(pos, max, ImGui::GetColorU32(color_rgb(100, 100, 100, 0.95f)), 4.0f);
  draw_list->AddText(font, font_size, ImVec2(pos.x + padding.x, pos.y + padding.y),
                     ImGui::GetColorU32(color_rgb(220, 220, 220)), label.c_str(), nullptr);
}

void pace_frame(const AppState &app, Clock::time_point frame_start) {
  const int fps = std::clamp(app.target_fps, kMinLoggyTargetFps, kMaxLoggyTargetFps);
  const Clock::duration period =
    std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(1.0 / static_cast<double>(fps)));
  const Clock::time_point wake_at = frame_start + period;
  if (Clock::now() < wake_at) std::this_thread::sleep_until(wake_at);
}

void draw_main_menu(AppState &app) {
  if (!ImGui::BeginMainMenuBar()) return;
  if (ImGui::BeginMenu("File")) {
    if (ImGui::MenuItem("Open Route...")) {
      request_route_popup(app.session, app.route_ui);
    }
    if (ImGui::MenuItem("Live Source...")) {
      request_live_source_popup(app.session, app.live_source_ui);
    }
    if (ImGui::MenuItem("Settings...")) {
      request_settings_popup(app.session, app.target_fps, app.settings_ui);
    }
    ImGui::Separator();
    if (ImGui::MenuItem("Quit")) app.request_close = true;
    ImGui::EndMenu();
  }
  if (ImGui::BeginMenu("Workspace")) {
    if (ImGui::MenuItem("Undo", nullptr, false, app.workspace_history.can_undo())) {
      const auto request = restore_workspace_snapshot(app.session.workspace, app.workspace_history.undo(),
                                                    app.session.workspace_layout_path, app.workspace_status,
                                                    "Undid workspace change");
      if (request.has_value()) app.workspace_select_request = request.value();
    }
    if (ImGui::MenuItem("Redo", nullptr, false, app.workspace_history.can_redo())) {
      const auto request = restore_workspace_snapshot(app.session.workspace, app.workspace_history.redo(),
                                                    app.session.workspace_layout_path, app.workspace_status,
                                                    "Redid workspace change");
      if (request.has_value()) app.workspace_select_request = request.value();
    }
    ImGui::Separator();
    if (ImGui::MenuItem("Save Layout", nullptr, false, workspace_autosave_available(app.session.workspace_layout_path))) {
      save_workspace_now(app.session.workspace, app.workspace_history, app.session.workspace_layout_path, app.workspace_status);
    }
    if (ImGui::MenuItem("Clear Draft", nullptr, false, workspace_autosave_available(app.session.workspace_layout_path))) {
      clear_workspace_draft_now(app.session.workspace_layout_path, app.workspace_status);
    }
    ImGui::EndMenu();
  }
  if (ImGui::BeginMenu("View")) {
    ImGui::MenuItem("Frame-Time HUD", nullptr, &app.show_frame_hud);
    ImGui::MenuItem("ImGui Demo", nullptr, &app.show_demo);
    ImGui::EndMenu();
  }
  if (ImGui::BeginMenu("Help")) {
    if (ImGui::MenuItem("About Loggy")) {
      request_help_popup(app);
    }
    ImGui::EndMenu();
  }
  ImGui::EndMainMenuBar();
}

std::string pane_window_label(const PaneInstance &pane, int pane_index) {
  const std::string title = pane.title.empty() ? kDefaultPaneTitle : pane.title;
  return title + "###pane_" + std::to_string(pane_index);
}

void request_split(AppState &app, int tab_index, int pane_index, PaneSplit split) {
  app.pending_pane_action = AppState::PaneAction{
    .type = AppState::PaneAction::Type::Split,
    .tab_index = tab_index,
    .pane_index = pane_index,
    .split = split,
  };
}

void draw_pane_surface(AppState &app, WorkspaceTab &tab, int tab_index, int pane_index, const ImVec2 &size) {
  if (pane_index < 0 || pane_index >= static_cast<int>(tab.panes.size())) return;
  PaneInstance &pane = tab.panes[static_cast<size_t>(pane_index)];
  ImGui::BeginChild(pane_window_label(pane, pane_index).c_str(), size, true);

  push_bold_font();
  ImGui::TextUnformatted(pane.title.empty() ? kDefaultPaneTitle : pane.title.c_str());
  pop_bold_font();
  ImGui::SameLine();
  ImGui::TextDisabled("%s", pane.type.c_str());

  if (ImGui::BeginPopupContextWindow()) {
    if (ImGui::MenuItem("Split Right")) request_split(app, tab_index, pane_index, PaneSplit::Right);
    if (ImGui::MenuItem("Split Bottom")) request_split(app, tab_index, pane_index, PaneSplit::Bottom);
    ImGui::Separator();
    if (ImGui::MenuItem("Close")) {
      app.pending_pane_action = AppState::PaneAction{
        .type = AppState::PaneAction::Type::Close,
        .tab_index = tab_index,
        .pane_index = pane_index,
      };
    }
    ImGui::EndPopup();
  }

  ImGui::Separator();
  if (const PaneType *type = pane_type(pane.type); type != nullptr && type->draw != nullptr) {
    type->draw(app.session, pane);
  }
  ImGui::EndChild();
}

void draw_workspace_node(AppState &app, WorkspaceTab &tab, int tab_index, const WorkspaceNode &node, ImVec2 pos, ImVec2 size) {
  size.x = std::max(1.0f, size.x);
  size.y = std::max(1.0f, size.y);
  if (node.is_pane) {
    ImGui::SetCursorScreenPos(pos);
    draw_pane_surface(app, tab, tab_index, node.pane_index, size);
    return;
  }
  if (node.children.empty()) return;

  const bool horizontal = node.orientation == SplitOrientation::Horizontal;
  const float gap = horizontal ? ImGui::GetStyle().ItemSpacing.x : ImGui::GetStyle().ItemSpacing.y;
  const float available = horizontal ? size.x : size.y;
  const float total_gap = gap * static_cast<float>(node.children.size() - 1);
  const float content = std::max(1.0f, available - total_gap);
  float offset = 0.0f;

  for (size_t i = 0; i < node.children.size(); ++i) {
    const float ratio = i < node.sizes.size() ? node.sizes[i] : 1.0f / static_cast<float>(node.children.size());
    ImVec2 child_size = size;
    ImVec2 child_pos = pos;
    if (horizontal) {
      child_size.x = std::max(1.0f, content * ratio);
      child_pos.x += offset;
      offset += child_size.x + gap;
    } else {
      child_size.y = std::max(1.0f, content * ratio);
      child_pos.y += offset;
      offset += child_size.y + gap;
    }
    ImGui::PushID(static_cast<int>(i));
    draw_workspace_node(app, tab, tab_index, node.children[i], child_pos, child_size);
    ImGui::PopID();
  }
  ImGui::SetCursorScreenPos(ImVec2(pos.x, pos.y + size.y));
  ImGui::Dummy(ImVec2(0.0f, 0.0f));
}

void apply_pending_pane_action(AppState &app) {
  if (!app.pending_pane_action.has_value()) return;
  const auto action = *app.pending_pane_action;
  app.pending_pane_action.reset();

  Workspace &workspace = app.session.workspace;
  if (action.tab_index < 0 || action.tab_index >= static_cast<int>(workspace.tabs.size())) return;
  WorkspaceTab &tab = workspace.tabs[static_cast<size_t>(action.tab_index)];
  if (action.type == AppState::PaneAction::Type::Close) {
    if (close_pane(&tab, action.pane_index)) {
      record_workspace_change(workspace, app.workspace_history, app.session.workspace_layout_path, app.workspace_status,
                             "Autosaved workspace draft");
    }
    return;
  }

  PaneInstance pane = make_pane("empty", "...");
  if (split_pane(&tab, action.pane_index, action.split, std::move(pane))) {
    record_workspace_change(app.session.workspace, app.workspace_history, app.session.workspace_layout_path,
                           app.workspace_status, "Autosaved workspace draft");
  }
}

void draw_workspace(AppState &app) {
  Workspace &workspace = app.session.workspace;
  normalize_workspace(&workspace);

  const int select_this_frame = app.workspace_select_request;
  app.workspace_select_request = -1;

  if (!ImGui::BeginTabBar("##workspace_tabs")) return;
  for (size_t i = 0; i < workspace.tabs.size(); ++i) {
    WorkspaceTab &tab = workspace.tabs[i];
    ImGuiTabItemFlags flags = select_this_frame == static_cast<int>(i) ? ImGuiTabItemFlags_SetSelected : ImGuiTabItemFlags_None;
    if (ImGui::BeginTabItem(tab.name.c_str(), nullptr, flags)) {
      workspace.current_tab_index = static_cast<int>(i);
      draw_workspace_node(app, tab, static_cast<int>(i), tab.root, ImGui::GetCursorScreenPos(), ImGui::GetContentRegionAvail());
      ImGui::EndTabItem();
    }
  }
  if (ImGui::TabItemButton("+", ImGuiTabItemFlags_Trailing | ImGuiTabItemFlags_NoTooltip)) {
    add_tab(&workspace);
    app.workspace_select_request = workspace.current_tab_index;
    record_workspace_change(workspace, app.workspace_history, app.session.workspace_layout_path,
                           app.workspace_status, "Autosaved workspace draft");
  }
  ImGui::EndTabBar();

  apply_pending_pane_action(app);
}

void draw_timeline_strip(AppState &app, const ImVec2 &size) {
  PlaybackClock &playback = app.session.playback;
  TimelineModel &timeline = app.session.timeline;
  const ImVec2 pos = ImGui::GetCursorScreenPos();
  const ImVec2 rect_max(pos.x + size.x, pos.y + size.y);
  ImDrawList *draw_list = ImGui::GetWindowDrawList();
  draw_list->AddRectFilled(pos, rect_max, ImGui::GetColorU32(color_rgb(47, 47, 47)), 3.0f);
  draw_list->AddRect(pos, rect_max, ImGui::GetColorU32(color_rgb(85, 85, 85)), 3.0f);

  for (const TimelineRenderSpan &span : timeline.render_spans()) {
    const float x0 = pos.x + static_cast<float>(span.start_fraction) * size.x;
    const float x1 = pos.x + static_cast<float>(span.end_fraction) * size.x;
    const TimelineColor c = span.color;
    draw_list->AddRectFilled(ImVec2(x0, pos.y + 1.0f), ImVec2(x1, rect_max.y - 1.0f),
                             IM_COL32(c.r, c.g, c.b, c.a), 2.0f);
  }

  const float tracker_x = pos.x + static_cast<float>(timeline.fraction_from_time(playback.tracker_time())) * size.x;
  draw_list->AddLine(ImVec2(tracker_x, pos.y - 2.0f), ImVec2(tracker_x, rect_max.y + 2.0f),
                     ImGui::GetColorU32(color_rgb(235, 235, 235)), 2.0f);

  ImGui::InvisibleButton("##timeline", size);
  if (ImGui::IsItemActive() && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
    const float fraction = size.x > 0.0f ? (ImGui::GetIO().MousePos.x - pos.x) / size.x : 0.0f;
    playback.seek(timeline.time_from_fraction(fraction));
  }
}

void draw_transport_bar(AppState &app) {
  PlaybackClock &playback = app.session.playback;
  if (ImGui::Button(playback.playing() ? "Pause" : "Play", ImVec2(64.0f, 0.0f))) {
    playback.toggle_playing();
  }
  ImGui::SameLine();
  if (ImGui::Button("-")) playback.step_backward();
  ImGui::SameLine();
  if (ImGui::Button("+")) playback.step_forward();
  ImGui::SameLine();

  double current_rate = playback.rate();
  char rate_label[32];
  std::snprintf(rate_label, sizeof(rate_label), "%.2gx", current_rate);
  ImGui::SetNextItemWidth(70.0f);
  if (ImGui::BeginCombo("##rate", rate_label)) {
    for (double rate : kPlaybackRatePresets) {
      const bool selected = std::abs(rate - current_rate) < 1.0e-9;
      char item_label[32];
      std::snprintf(item_label, sizeof(item_label), "%.2gx", rate);
      if (ImGui::Selectable(item_label, selected)) {
        playback.set_rate(rate);
      }
      if (selected) ImGui::SetItemDefaultFocus();
    }
    ImGui::EndCombo();
  }
  ImGui::SameLine();
  bool loop = playback.loop();
  if (ImGui::Checkbox("Loop", &loop)) {
    playback.set_loop(loop);
  }
}

void draw_shell(AppState &app) {
  draw_main_menu(app);

  const ImGuiViewport *viewport = ImGui::GetMainViewport();
  const float menu_height = ImGui::GetFrameHeight();
  const float status_height = 76.0f;
  const ImVec2 host_pos(viewport->WorkPos.x, viewport->WorkPos.y + menu_height);
  const ImVec2 host_size(viewport->WorkSize.x, std::max(0.0f, viewport->WorkSize.y - menu_height - status_height));

  ImGui::SetNextWindowPos(host_pos);
  ImGui::SetNextWindowSize(host_size);
  ImGui::Begin("Workspace", nullptr, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove |
                                      ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings |
                                      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
  draw_workspace(app);
  ImGui::End();

  if (app.show_demo) {
    ImGui::ShowDemoWindow(&app.show_demo);
  }

  ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x, viewport->WorkPos.y + viewport->WorkSize.y - status_height));
  ImGui::SetNextWindowSize(ImVec2(viewport->WorkSize.x, status_height));
  ImGui::Begin("##loggy_status_bar", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                                      ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings);
  draw_transport_bar(app);
  ImGui::SameLine();
  ImGui::Text("  %.2fs / %.2fs", app.session.playback.tracker_time(), app.session.playback.route_range().end);
  draw_timeline_strip(app, ImVec2(ImGui::GetContentRegionAvail().x, 12.0f));
  const RouteIngestStatus ingest = app.session.ingest_status();
  const LivePollSnapshot live = app.session.live_status();
  const SessionConfig &session_config = app.session.config;
  const bool stream = session_config.stream;
  const RouteSelection route_selection = current_route_selection(app.session);
  const std::string route_spec = route_selection_full_spec(route_selection);
  ImGui::Text("%s  |  %s preset", route_ingest_state_label(ingest.state), app.options.preset.c_str());
  ImGui::SameLine();
  if (stream) {
    ImGui::TextDisabled(" | %s %s", live_source_kind_label(session_config.stream_source_kind),
                        live.source_label.c_str());
    ImGui::SameLine();
    if (ImGui::SmallButton("Source")) {
      request_live_source_popup(app.session, app.live_source_ui);
    }
    ImGui::SameLine();
    if (ImGui::SmallButton(live.paused ? "Resume Live" : "Pause Live")) {
      app.session.set_live_paused(!live.paused);
    }
    ImGui::SameLine();
    bool live_follow = app.session.live_follow;
    if (ImGui::Checkbox("Follow live", &live_follow)) {
      app.session.set_live_follow(live_follow);
    }
    ImGui::SameLine();
    ImGui::TextDisabled("  |  live %s  %llu msgs  %llu batches  %.0fs buffer  %zu series  %zu CAN ids",
                        live.paused ? "paused" : (live.connected ? "connected" : (live.active ? "listening" : "stopped")),
                        static_cast<unsigned long long>(live.received_messages),
                        static_cast<unsigned long long>(live.published_batches),
                        live.buffer_seconds,
                        app.session.store.series_path_count(), app.session.store.can_message_count());
    if (!live.error.empty()) {
      ImGui::SameLine();
      ImGui::TextDisabled("  |  %s", live.error.c_str());
    }
  } else {
    if (ImGui::SmallButton("Route")) {
      request_route_popup(app.session, app.route_ui);
    }
    ImGui::SameLine();
    ImGui::TextDisabled("  |  %s  |  %s  |  %zu/%zu segments  %zu series  %zu CAN ids",
                        route_spec.empty() ? (session_config.route_name.empty() ? "no route" : session_config.route_name.c_str()) : route_spec.c_str(),
                        log_selector_description(route_selection.selector),
                        ingest.segments_loaded, ingest.segments_resolved,
                        app.session.store.series_path_count(), app.session.store.can_message_count());
    if (session_config.route_name.empty()) {
      ImGui::SameLine();
      if (ImGui::SmallButton("Open Live")) {
        request_live_source_popup(app.session, app.live_source_ui);
      }
    }
  }
  if (!app.workspace_status.empty()) {
    ImGui::SameLine();
    ImGui::TextDisabled("  |  %s", app.workspace_status.c_str());
  }
  ImGui::End();

  draw_route_popup(app.session, app.route_ui, modal_escape_pressed());
  RemoteRouteBrowserActions remote_route_browser_actions;
  remote_route_browser_actions.ctx = &app;
  remote_route_browser_actions.open_route = [](void *ctx, std::string_view route) {
    AppState &target = *static_cast<AppState *>(ctx);
    const std::string route_name(route);
    std::snprintf(target.route_ui.route_name_buffer.data(), target.route_ui.route_name_buffer.size(), "%s",
                  route_name.c_str());
    restart_route_from_popup(target.session, target.route_ui, route_name, "Opened remote route");
  };
  draw_remote_route_browser(remote_route_browser_actions);
  draw_live_source_popup(app.session, app.live_source_ui);
  draw_settings_popup(app.session, modal_escape_pressed(), app.options.show_frame_hud, app.theme_kind, app.target_fps,
                     app.show_frame_hud, app.settings_ui);
  draw_help_popup(app);

  if (app.show_frame_hud) {
    draw_frame_hud(app.frame_stats);
  }
}

void render_frame(GLFWwindow *window, AppState &app, const fs::path *capture_path) {
  const auto frame_start = Clock::now();
  glfwPollEvents();

  int framebuffer_width = 0;
  int framebuffer_height = 0;
  glfwGetFramebufferSize(window, &framebuffer_width, &framebuffer_height);

  const auto cpu_start = Clock::now();
  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  if (!ImGui::GetIO().WantTextInput && ImGui::IsKeyPressed(ImGuiKey_F12, false)) {
    app.show_frame_hud = !app.show_frame_hud;
  }
  if (!ImGui::GetIO().WantTextInput && ImGui::IsKeyPressed(ImGuiKey_Space, false)) {
    app.session.playback.toggle_playing();
  }
  if (!ImGui::GetIO().WantTextInput && ImGui::IsKeyPressed(ImGuiKey_F1, false)) {
    request_help_popup(app);
  }
  // Cabana-style frame stepping; repeat stays on since steps are cheap.
  if (!ImGui::GetIO().WantTextInput && ImGui::IsKeyPressed(ImGuiKey_LeftArrow)) app.session.playback.step_backward();
  if (!ImGui::GetIO().WantTextInput && ImGui::IsKeyPressed(ImGuiKey_RightArrow)) app.session.playback.step_forward();

  const auto playback_now = Clock::now();
  const double playback_dt = std::chrono::duration<double>(playback_now - app.last_playback_update).count();
  app.last_playback_update = playback_now;
  app.session.playback.advance(playback_dt);
  app.session.begin_frame();

  draw_shell(app);
  g_escape_pressed = false;
  ImGui::Render();

  const ImVec4 clear = clear_color();
  glViewport(0, 0, framebuffer_width, framebuffer_height);
  glClearColor(clear.x, clear.y, clear.z, clear.w);
  glClear(GL_COLOR_BUFFER_BIT);
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
  const auto cpu_end = Clock::now();
  app.frame_stats.add(std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count());

  if (capture_path != nullptr) {
    save_framebuffer_png(*capture_path, framebuffer_width, framebuffer_height);
  }
  glfwSwapBuffers(window);
  pace_frame(app, frame_start);

  if (app.request_close) {
    glfwSetWindowShouldClose(window, GLFW_TRUE);
    app.request_close = false;
  }
}

}  // namespace

int run(const Options &options) {
  try {
    ShutdownSignalHandlers shutdown_handlers;
    GlfwRuntime glfw_runtime(options);
    ImGuiRuntime imgui_runtime(glfw_runtime.window());
    AppState app(options);

    if (!options.output_path.empty()) {
      for (int i = 0; i < 4; ++i) {
        if (shutdown_requested()) return 0;
        render_frame(glfw_runtime.window(), app, nullptr);
      }
      app.frame_stats.reset();
      if (shutdown_requested()) return 0;
      render_frame(glfw_runtime.window(), app, nullptr);
      const fs::path capture_path = options.output_path;
      if (shutdown_requested()) return 0;
      render_frame(glfw_runtime.window(), app, &capture_path);
      return 0;
    }

    while (!glfwWindowShouldClose(glfw_runtime.window()) && !shutdown_requested()) {
      render_frame(glfw_runtime.window(), app, nullptr);
    }
    // Say why we exited: a silent 0 on a stray SIGTERM is indistinguishable from a crash.
    if (shutdown_requested()) std::cerr << "loggy: exiting on signal " << g_shutdown_signal << "\n";
    return 0;
  } catch (const std::exception &err) {
    std::cerr << "loggy: " << err.what() << "\n";
    return 1;
  }
}

}  // namespace loggy
