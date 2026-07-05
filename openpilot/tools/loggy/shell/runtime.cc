#include "tools/loggy/shell/runtime.h"

#include "tools/loggy/backend/session.h"
#include "tools/loggy/shell/theme.h"
#include "tools/loggy/shell/workspace.h"

#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_opengl3_loader.h"
#include "implot.h"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace loggy {
namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

std::string shell_quote(std::string_view value) {
  std::string quoted;
  quoted.reserve(value.size() + 8);
  quoted.push_back('\'');
  for (char c : value) {
    if (c == '\'') {
      quoted += "'\\''";
    } else {
      quoted.push_back(c);
    }
  }
  quoted.push_back('\'');
  return quoted;
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
  static constexpr size_t kWindow = 240;
  std::array<double, kWindow> samples{};
  size_t cursor = 0;
  size_t count = 0;
  double latest_ms = 0.0;

  void add(double ms) {
    latest_ms = ms;
    samples[cursor] = ms;
    cursor = (cursor + 1) % samples.size();
    count = std::min(count + 1, samples.size());
  }

  void reset() {
    cursor = 0;
    count = 0;
    latest_ms = 0.0;
    samples.fill(0.0);
  }

  double p99_ms() const {
    if (count == 0) return 0.0;
    std::vector<double> values(samples.begin(), samples.begin() + static_cast<ptrdiff_t>(count));
    std::sort(values.begin(), values.end());
    const size_t idx = std::min(values.size() - 1, static_cast<size_t>(values.size() * 0.99));
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
        .stream = opts.stream,
      }),
      show_frame_hud(opts.show_frame_hud) {}

  const Options &options;
  Session session;
  FrameStats frame_stats;
  Clock::time_point last_playback_update = Clock::now();
  bool show_frame_hud = true;
  bool show_demo = false;
  bool request_close = false;

  struct PaneAction {
    enum class Type { Split, Close };
    Type type = Type::Split;
    int tab_index = -1;
    int pane_index = -1;
    PaneSplit split = PaneSplit::Right;
  };
  std::optional<PaneAction> pending_pane_action;
};

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

void draw_main_menu(AppState &app) {
  if (!ImGui::BeginMainMenuBar()) return;
  if (ImGui::BeginMenu("File")) {
    ImGui::MenuItem("Open Route...", nullptr, false, false);
    ImGui::Separator();
    if (ImGui::MenuItem("Quit")) app.request_close = true;
    ImGui::EndMenu();
  }
  if (ImGui::BeginMenu("View")) {
    ImGui::MenuItem("Frame-Time HUD", nullptr, &app.show_frame_hud);
    ImGui::MenuItem("ImGui Demo", nullptr, &app.show_demo);
    ImGui::EndMenu();
  }
  if (ImGui::BeginMenu("Help")) {
    ImGui::MenuItem("About Loggy", nullptr, false, false);
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
  if (const PaneType *type = pane_registry().find(pane.type); type != nullptr && type->draw) {
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
}

void apply_pending_pane_action(AppState &app) {
  if (!app.pending_pane_action.has_value()) return;
  const auto action = *app.pending_pane_action;
  app.pending_pane_action.reset();

  Workspace &workspace = app.session.workspace();
  if (action.tab_index < 0 || action.tab_index >= static_cast<int>(workspace.tabs.size())) return;
  WorkspaceTab &tab = workspace.tabs[static_cast<size_t>(action.tab_index)];
  if (action.type == AppState::PaneAction::Type::Close) {
    close_pane(&tab, action.pane_index);
    return;
  }

  PaneInstance pane = make_pane("empty", "...");
  split_pane(&tab, action.pane_index, action.split, std::move(pane));
}

void draw_workspace(AppState &app) {
  Workspace &workspace = app.session.workspace();
  normalize_workspace(&workspace);

  static Workspace *selection_workspace = nullptr;
  static int select_request = -1;
  if (selection_workspace != &workspace) {
    selection_workspace = &workspace;
    select_request = workspace.current_tab_index;
  }
  const int select_this_frame = select_request;
  select_request = -1;

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
    select_request = workspace.current_tab_index;
  }
  ImGui::EndTabBar();

  apply_pending_pane_action(app);
}

void draw_timeline_strip(AppState &app, const ImVec2 &size) {
  PlaybackClock &playback = app.session.playback();
  TimelineModel &timeline = app.session.timeline();
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
  PlaybackClock &playback = app.session.playback();
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
  ImGui::Text("  %.2fs / %.2fs", app.session.playback().tracker_time(), app.session.playback().route_range().end);
  draw_timeline_strip(app, ImVec2(ImGui::GetContentRegionAvail().x, 12.0f));
  const RouteIngestStatus ingest = app.session.ingestStatus();
  ImGui::Text("%s  |  %s preset  |  %s", routeIngestStateLabel(ingest.state), app.options.preset.c_str(),
              app.options.stream ? app.options.stream_address.c_str()
                                 : (app.options.route_name.empty() ? "no route" : app.options.route_name.c_str()));
  ImGui::SameLine();
  ImGui::TextDisabled("  |  %zu/%zu segments  %zu series  %zu CAN ids",
                      ingest.segments_loaded, ingest.segments_resolved,
                      app.session.store().seriesPathCount(), app.session.store().canMessageCount());
  ImGui::End();

  if (app.show_frame_hud) {
    draw_frame_hud(app.frame_stats);
  }
}

void render_frame(GLFWwindow *window, AppState &app, const fs::path *capture_path) {
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
    app.session.playback().toggle_playing();
  }

  const auto playback_now = Clock::now();
  const double playback_dt = std::chrono::duration<double>(playback_now - app.last_playback_update).count();
  app.last_playback_update = playback_now;
  app.session.playback().advance(playback_dt);
  app.session.beginFrame();

  draw_shell(app);
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

  if (app.request_close) {
    glfwSetWindowShouldClose(window, GLFW_TRUE);
    app.request_close = false;
  }
}

}  // namespace

int run(const Options &options) {
  try {
    GlfwRuntime glfw_runtime(options);
    ImGuiRuntime imgui_runtime(glfw_runtime.window());
    AppState app(options);

    if (!options.output_path.empty()) {
      for (int i = 0; i < 4; ++i) {
        render_frame(glfw_runtime.window(), app, nullptr);
      }
      app.frame_stats.reset();
      render_frame(glfw_runtime.window(), app, nullptr);
      const fs::path capture_path = options.output_path;
      render_frame(glfw_runtime.window(), app, &capture_path);
      return 0;
    }

    while (!glfwWindowShouldClose(glfw_runtime.window())) {
      render_frame(glfw_runtime.window(), app, nullptr);
    }
    return 0;
  } catch (const std::exception &err) {
    std::cerr << "loggy: " << err.what() << "\n";
    return 1;
  }
}

}  // namespace loggy
