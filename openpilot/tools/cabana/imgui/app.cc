#include "tools/cabana/imgui/app.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <string>

#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_opengl3_loader.h"
#include "imgui_internal.h"

#include <GLFW/glfw3.h>

#include "tools/cabana/imgui/dbc_menus.h"
#include "tools/cabana/settings.h"

namespace fs = std::filesystem;

namespace {

constexpr float MESSAGES_WIDTH_RATIO = 0.30f;
constexpr float RIGHT_COLUMN_WIDTH_RATIO = 0.27f;
constexpr float VIDEO_HEIGHT_RATIO = 0.45f;

fs::path config_dir() {
  if (const char *xdg = std::getenv("XDG_CONFIG_HOME"); xdg != nullptr && xdg[0] != '\0') {
    return fs::path(xdg) / "cabana";
  }
  const char *home = std::getenv("HOME");
  return fs::path(home != nullptr ? home : "") / ".config" / "cabana";
}

// Single AppState instance for the process lifetime; used by the GLFW
// window-close callback below, which (per GLFW's C API) can't capture state.
AppState *g_app_for_close = nullptr;

// Intercept the OS-level close button (title bar X / Alt+F4 / window manager
// close) so it goes through the same DBC-aware "Unsaved Changes" flow as
// Ctrl+Q instead of closing immediately -- mirrors MainWindow::closeEvent()
// calling remindSaveChanges() before accepting a Qt close event.
void on_glfw_window_close(GLFWwindow *window) {
  glfwSetWindowShouldClose(window, GLFW_FALSE);
  if (g_app_for_close != nullptr) g_app_for_close->request_close = true;
}

void draw_menu_bar(AppState &app) {
  if (!ImGui::BeginMainMenuBar()) return;

  if (ImGui::BeginMenu("File")) {
    if (ImGui::MenuItem("Open Stream...")) open_stream_selector();
    ImGui::Separator();
    draw_dbc_file_menu_items(app);
    ImGui::Separator();
    if (ImGui::MenuItem("Export to CSV...", nullptr, false, has_stream(app))) open_export_csv();
    ImGui::Separator();
    if (ImGui::MenuItem("Exit", "Ctrl+Q")) app.request_close = true;
    ImGui::EndMenu();
  }
  if (ImGui::BeginMenu("Edit")) {
    draw_dbc_edit_menu_items();
    ImGui::Separator();
    if (ImGui::MenuItem("Settings...", "Ctrl+,")) open_settings_dialog();
    ImGui::EndMenu();
  }
  if (ImGui::BeginMenu("View")) {
    const bool dark = app.theme == Theme::Dark;
    if (ImGui::MenuItem("Dark Theme", nullptr, dark)) {
      app.theme = dark ? Theme::Light : Theme::Dark;
      app.theme_changed = true;
      settings.theme = dark ? LIGHT_THEME : DARK_THEME;
      settings.changed();
    }
    ImGui::MenuItem("Full Screen", "F11", false, false);
    ImGui::EndMenu();
  }
  if (ImGui::BeginMenu("Tools")) {
    if (ImGui::MenuItem("Find Signal...", nullptr, false, has_stream(app))) open_find_signal();
    if (ImGui::MenuItem("Find Similar Bits...", nullptr, false, has_stream(app))) open_find_similar_bits();
    ImGui::EndMenu();
  }
  if (ImGui::BeginMenu("Help")) {
    if (ImGui::MenuItem("Help", "F1")) toggle_help_overlay();
    if (ImGui::MenuItem("About")) app.show_about = true;
    ImGui::EndMenu();
  }
  ImGui::EndMainMenuBar();
}

// First run (no persisted imgui.ini dock node): lay out the shell like Qt
// cabana's mainwin.cc createDockWindows()/createDockWidgets() -- Messages on
// the left, Video/Charts stacked on the right, detail view in the center.
void build_default_layout(ImGuiID dockspace_id, const ImVec2 &size) {
  ImGui::DockBuilderAddNode(dockspace_id, ImGuiDockNodeFlags_DockSpace);
  ImGui::DockBuilderSetNodeSize(dockspace_id, size);

  ImGuiID left_id = 0, remainder_id = 0, right_id = 0, center_id = 0, video_id = 0, charts_id = 0;
  ImGui::DockBuilderSplitNode(dockspace_id, ImGuiDir_Left, MESSAGES_WIDTH_RATIO, &left_id, &remainder_id);
  const float right_ratio = RIGHT_COLUMN_WIDTH_RATIO / (1.0f - MESSAGES_WIDTH_RATIO);
  ImGui::DockBuilderSplitNode(remainder_id, ImGuiDir_Right, right_ratio, &right_id, &center_id);
  ImGui::DockBuilderSplitNode(right_id, ImGuiDir_Up, VIDEO_HEIGHT_RATIO, &video_id, &charts_id);

  ImGui::DockBuilderDockWindow(MESSAGES_WINDOW_TITLE, left_id);
  ImGui::DockBuilderDockWindow(VIDEO_WINDOW_TITLE, video_id);
  ImGui::DockBuilderDockWindow(CHARTS_WINDOW_TITLE, charts_id);
  ImGui::DockBuilderDockWindow(CENTER_WINDOW_TITLE, center_id);

  ImGui::DockBuilderFinish(dockspace_id);
}

void draw_dockspace(AppState &app, const ImVec2 &pos, const ImVec2 &size) {
  ImGui::SetNextWindowPos(pos);
  ImGui::SetNextWindowSize(size);
  const ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                                 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings |
                                 ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse;
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
  ImGui::Begin("##dockspace_host", nullptr, flags);
  ImGui::PopStyleVar();

  const ImGuiID dockspace_id = ImGui::GetID("CabanaDockspace");
  if (ImGui::DockBuilderGetNode(dockspace_id) == nullptr) {
    build_default_layout(dockspace_id, size);
  }
  ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_AutoHideTabBar);
  ImGui::End();

  draw_messages_panel(app);
  draw_video_panel(app);
  draw_charts_panel(app);
  draw_detail_panel(app);
}

void draw_about_popup(AppState &app) {
  if (app.show_about) {
    ImGui::OpenPopup("About Cabana");
    app.show_about = false;
  }
  ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
  if (ImGui::BeginPopupModal("About Cabana", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    push_bold_font();
    ImGui::TextUnformatted("Cabana");
    pop_bold_font();
    ImGui::TextUnformatted("A CAN bus analysis tool.");
    ImGui::TextUnformatted("https://github.com/commaai/openpilot/tree/master/openpilot/tools/cabana");
    ImGui::Spacing();
    if (ImGui::Button("Close") || ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  }
}

void draw_ui(AppState &app) {
  dbc_menus_update();

  const ImGuiIO &io = ImGui::GetIO();
  const bool ctrl = io.KeyCtrl || io.KeySuper;
  if (!io.WantTextInput && ctrl && ImGui::IsKeyPressed(ImGuiKey_Q, false)) {
    app.request_close = true;
  }
  if (!io.WantTextInput && ImGui::IsKeyPressed(ImGuiKey_Space, false)) {
    app.stream->pause(!app.stream->isPaused());
  }
  if (!io.WantTextInput && ImGui::IsKeyPressed(ImGuiKey_F1, false)) {
    toggle_help_overlay();
  }

  // DBC-aware close: consume the trigger edge here (both Ctrl+Q above and
  // the GLFW window-close callback below set this) and route it through the
  // "Unsaved Changes" reminder instead of closing immediately. dbc_menus.cc
  // calls glfwSetWindowShouldClose() itself once the flow resolves.
  if (app.request_close) {
    app.request_close = false;
    dbc_menus_begin_close();
  }

  draw_menu_bar(app);

  const ImGuiViewport *viewport = ImGui::GetMainViewport();
  const ImVec2 work_pos = viewport->WorkPos;
  const ImVec2 work_size = viewport->WorkSize;
  const float content_height = work_size.y - TRANSPORT_BAR_HEIGHT;
  draw_dockspace(app, work_pos, ImVec2(work_size.x, content_height));
  draw_transport_bar(app);
  draw_about_popup(app);
  draw_dbc_modals();
  draw_stream_selector(app);
  draw_find_signal(app);
  draw_find_similar_bits(app);
  draw_settings_dialog(app);
  draw_route_tools(app);
  draw_help_overlay(app);
}

void render_frame(GLFWwindow *window, AppState &app, const fs::path *capture_path) {
  glfwPollEvents();

  if (app.theme_changed) {
    apply_theme(app.theme);
    app.theme_changed = false;
  }

  int framebuffer_width = 0;
  int framebuffer_height = 0;
  glfwGetFramebufferSize(window, &framebuffer_width, &framebuffer_height);

  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  app.stream->update();

  draw_ui(app);

  ImGui::Render();

  glViewport(0, 0, framebuffer_width, framebuffer_height);
  const ImVec4 clear_color = theme_clear_color();
  glClearColor(clear_color.x, clear_color.y, clear_color.z, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT);
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
  if (capture_path != nullptr) {
    save_framebuffer_png(*capture_path, framebuffer_width, framebuffer_height);
  }
  glfwSwapBuffers(window);
}

}  // namespace

int run(const Options &options, std::unique_ptr<AbstractStream> stream) {
  const fs::path settings_dir = config_dir();
  std::error_code ec;
  fs::create_directories(settings_dir, ec);
  const std::string ini_path = (settings_dir / "imgui.ini").string();

  GlfwRuntime glfw_runtime(options);
  ImGuiRuntime imgui_runtime(glfw_runtime.window());
  ImGui::GetIO().IniFilename = ini_path.c_str();
  load_fonts();

  AppState app;
  app.stream = std::move(stream);
  can = app.stream.get();

  app.theme = theme_from_settings();
  if (options.dark_theme.has_value()) {
    app.theme = *options.dark_theme ? Theme::Dark : Theme::Light;
    settings.theme = *options.dark_theme ? DARK_THEME : LIGHT_THEME;
  }
  apply_theme(app.theme);

  dbc_menus_init(glfw_runtime.window());
  g_app_for_close = &app;
  glfwSetWindowCloseCallback(glfw_runtime.window(), on_glfw_window_close);

  app.stream->start();
  dbc_menus_ensure_dbc_open();  // mirrors startStream()'s "don't leave zero DBCs loaded"

  if (!options.output_path.empty()) {
    const fs::path capture_path = options.output_path;
    for (int i = 0; i < 3; ++i) {
      render_frame(glfw_runtime.window(), app, nullptr);
    }
    if (has_stream(app)) {
      // Pump real frames so the capture is data-bearing instead of a route
      // that never advanced past t=0.
      const double target_sec = std::min(30.0, app.stream->maxSeconds() * 0.5);
      app.stream->setSpeed(5.0f);
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(90);
      while (app.stream->currentSec() < target_sec && std::chrono::steady_clock::now() < deadline) {
        render_frame(glfw_runtime.window(), app, nullptr);
      }
      app.stream->setSpeed(1.0f);
      if (!app.selected_msg_id && !can->lastMessages().empty()) {
        // select the busiest message so panel captures are populated
        auto busiest = std::max_element(can->lastMessages().begin(), can->lastMessages().end(),
                                        [](const auto &a, const auto &b) { return a.second.count < b.second.count; });
        app.selected_msg_id = busiest->first;
      }
    }
    render_frame(glfw_runtime.window(), app, &capture_path);
    if (!options.show) return 0;
  }

  while (!glfwWindowShouldClose(glfw_runtime.window())) {
    render_frame(glfw_runtime.window(), app, nullptr);
  }
  return 0;
}
