#include "tools/jotpluggler/internal.h"

#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_opengl3_loader.h"

#include <GLFW/glfw3.h>

namespace fs = std::filesystem;

void draw_fps_overlay(const UiState &state, float top_offset) {
  if (!state.show_fps_overlay) {
    return;
  }
  ImGuiViewport *viewport = ImGui::GetMainViewport();
  const ImGuiIO &io = ImGui::GetIO();
  const float fps = io.Framerate;
  const std::string label = util::string_format("%.1f fps", fps);

  const ImVec2 padding(10.0f, 8.0f);
  const ImVec2 margin(12.0f, 10.0f);
  app_push_mono_font();
  ImFont *font = ImGui::GetFont();
  const float font_size = ImGui::GetFontSize();
  const ImVec2 text_size = ImGui::CalcTextSize(label.c_str());
  app_pop_mono_font();
  const ImVec2 size(text_size.x + padding.x * 2.0f, text_size.y + padding.y * 2.0f);
  const ImVec2 pos(viewport->Pos.x + viewport->Size.x - size.x - margin.x,
                   viewport->Pos.y + top_offset + margin.y);
  ImDrawList *draw_list = ImGui::GetForegroundDrawList(viewport);
  const ImVec2 max(pos.x + size.x, pos.y + size.y);
  draw_list->AddRectFilled(pos, max, ImGui::GetColorU32(color_rgb(248, 249, 251, 0.92f)), 4.0f);
  draw_list->AddRect(pos, max, ImGui::GetColorU32(color_rgb(182, 188, 196, 0.95f)), 4.0f);
  draw_list->AddText(font, font_size, ImVec2(pos.x + padding.x, pos.y + padding.y),
                     ImGui::GetColorU32(color_rgb(57, 62, 69)), label.c_str(), nullptr);
}

void render_layout(AppSession *session, UiState *state, bool show_camera_feed) {
  if (!state->fps_overlay_initialized) {
    state->show_fps_overlay = false;
    state->fps_overlay_initialized = true;
  }
  ensure_shared_range(state, *session);
  if (state->follow_latest) {
    update_follow_range(state, *session);
    state->suppress_range_side_effects = true;
  } else {
    clamp_shared_range(state, *session);
  }
  const bool ctrl = ImGui::GetIO().KeyCtrl || ImGui::GetIO().KeySuper;
  const bool shift = ImGui::GetIO().KeyShift;
  if (!ImGui::GetIO().WantTextInput && ctrl && ImGui::IsKeyPressed(ImGuiKey_Z, false)) {
    if (shift) {
      apply_redo(session, state);
    } else {
      apply_undo(session, state);
    }
  }
  if (!ImGui::GetIO().WantTextInput && ctrl && ImGui::IsKeyPressed(ImGuiKey_F, false)) {
    state->open_find_signal = true;
  }
  if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow, false)) {
    step_tracker(state, -1.0);
  }
  if (ImGui::IsKeyPressed(ImGuiKey_RightArrow, false)) {
    step_tracker(state, 1.0);
  }
  if (!ImGui::GetIO().WantTextInput && ImGui::IsKeyPressed(ImGuiKey_Space, false)) {
    state->playback_playing = !state->playback_playing;
  }
  advance_playback(state, *session);
  CameraFeedView *sidebar_camera = session->pane_camera_feeds[static_cast<size_t>(sidebar_preview_camera_view(*session))].get();
  if (show_camera_feed && sidebar_camera != nullptr && state->has_tracker_time) {
    sidebar_camera->update(state->tracker_time);
  }
  const float menu_height = draw_main_menu_bar(session, state);
  UiMetrics ui = compute_ui_metrics(ImGui::GetMainViewport()->Size, menu_height, state->sidebar_width);
  if (state->browser_nodes_dirty) {
    rebuild_browser_nodes(session, state);
    state->browser_nodes_dirty = false;
  }
  state->sidebar_width = ui.sidebar_width;
  draw_sidebar(session, ui, state, show_camera_feed);
  draw_workspace(session, ui, state);
  draw_sidebar_resizer(ui, state);
  if (!state->custom_series.selected && !state->logs.selected) {
    draw_pane_windows(session, state);
  }
  draw_status_bar(*session, ui, state);
  draw_popups(session, state);
  draw_fps_overlay(*state, menu_height);
}

void save_framebuffer_png(const fs::path &output_path, int width, int height) {
  ensure_parent_dir(output_path);
  if (width <= 0 || height <= 0) throw std::runtime_error("Invalid framebuffer size");

  std::vector<uint8_t> pixels(static_cast<size_t>(width) * static_cast<size_t>(height) * 4U, 0);
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());

  const fs::path ppm_path = output_path.parent_path() / (output_path.stem().string() + ".ppm");
  std::string ppm = util::string_format("P6\n%d %d\n255\n", width, height);
  ppm.reserve(ppm.size() + static_cast<size_t>(width) * static_cast<size_t>(height) * 3U);
  for (int y = height - 1; y >= 0; --y) {
    for (int x = 0; x < width; ++x) {
      const size_t index = static_cast<size_t>((y * width + x) * 4);
      ppm.append(reinterpret_cast<const char *>(&pixels[index]), 3);
    }
  }
  write_file_or_throw(ppm_path, ppm.data(), ppm.size());

  const std::string command = "convert " + shell_quote(ppm_path.string()) + " " + shell_quote(output_path.string());
  run_system_or_throw(command, "image conversion");
  fs::remove(ppm_path);
}

void render_frame(GLFWwindow *window, AppSession *session, UiState *state, const fs::path *capture_path) {
  glfwPollEvents();

  int framebuffer_width = 0;
  int framebuffer_height = 0;
  glfwGetFramebufferSize(window, &framebuffer_width, &framebuffer_height);

  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  if (state->request_save_layout) {
    if (session->layout_path.empty()) {
      state->open_save_layout = true;
    } else {
      save_layout(session, state, session->layout_path.string());
    }
    state->request_save_layout = false;
  }
  if (state->request_reset_layout) {
    reset_layout(session, state);
    state->request_reset_layout = false;
  }
  poll_async_route_load(session, state);
  if (session->data_mode == SessionDataMode::Stream && session->stream_poller) {
    StreamExtractBatch batch;
    std::string error_text;
    if (session->stream_poller->consume(&batch, &error_text)) {
      if (!error_text.empty()) {
        state->error_text = error_text;
        state->open_error_popup = true;
        state->status_text = "Stream disconnected";
      } else {
        apply_stream_batch(session, state, std::move(batch));
      }
    }
  }

  const bool show_camera = capture_path == nullptr && session->data_mode != SessionDataMode::Stream;
  render_layout(session, state, show_camera);
  ImGui::Render();
  if (state->request_close) {
    glfwSetWindowShouldClose(window, GLFW_TRUE);
    state->request_close = false;
  }

  glViewport(0, 0, framebuffer_width, framebuffer_height);
  glClearColor(227.0f / 255.0f, 229.0f / 255.0f, 233.0f / 255.0f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT);
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
  if (capture_path != nullptr) {
    save_framebuffer_png(*capture_path, framebuffer_width, framebuffer_height);
  }
  glfwSwapBuffers(window);
  state->suppress_range_side_effects = false;
}
