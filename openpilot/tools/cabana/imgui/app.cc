#include "tools/cabana/imgui/app.h"

#include <string>

#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_opengl3_loader.h"

#include <GLFW/glfw3.h>

namespace fs = std::filesystem;

namespace {

constexpr float STATUS_BAR_HEIGHT = 28.0f;

struct AppState {
  Theme theme = Theme::Light;
  bool theme_changed = false;
  bool show_about = false;
  bool request_close = false;
  std::string status_text = "no stream";
};

void draw_menu_bar(AppState *state) {
  if (!ImGui::BeginMainMenuBar()) return;

  if (ImGui::BeginMenu("File")) {
    ImGui::MenuItem("Open Stream...", nullptr, false, false);
    ImGui::Separator();
    ImGui::MenuItem("New DBC File", "Ctrl+N", false, false);
    ImGui::MenuItem("Open DBC File...", "Ctrl+O", false, false);
    ImGui::MenuItem("Open Recent", nullptr, false, false);
    ImGui::MenuItem("Load DBC from commaai/opendbc", nullptr, false, false);
    ImGui::Separator();
    ImGui::MenuItem("Save DBC...", "Ctrl+S", false, false);
    ImGui::Separator();
    if (ImGui::MenuItem("Exit", "Ctrl+Q")) state->request_close = true;
    ImGui::EndMenu();
  }
  if (ImGui::BeginMenu("Edit")) {
    ImGui::MenuItem("Undo", "Ctrl+Z", false, false);
    ImGui::MenuItem("Redo", "Ctrl+Shift+Z", false, false);
    ImGui::Separator();
    ImGui::MenuItem("Settings...", "Ctrl+,", false, false);
    ImGui::EndMenu();
  }
  if (ImGui::BeginMenu("View")) {
    const bool dark = state->theme == Theme::Dark;
    if (ImGui::MenuItem("Dark Theme", nullptr, dark)) {
      state->theme = dark ? Theme::Light : Theme::Dark;
      state->theme_changed = true;
    }
    ImGui::MenuItem("Full Screen", "F11", false, false);
    ImGui::EndMenu();
  }
  if (ImGui::BeginMenu("Tools")) {
    ImGui::MenuItem("Find Signal...", nullptr, false, false);
    ImGui::MenuItem("Find Similar Bits...", nullptr, false, false);
    ImGui::EndMenu();
  }
  if (ImGui::BeginMenu("Help")) {
    ImGui::MenuItem("Help", "F1", false, false);
    if (ImGui::MenuItem("About")) state->show_about = true;
    ImGui::EndMenu();
  }
  ImGui::EndMainMenuBar();
}

void draw_centered_text(const char *text) {
  const float width = ImGui::GetContentRegionAvail().x;
  ImGui::SetCursorPosX((width - ImGui::CalcTextSize(text).x) * 0.5f);
  ImGui::TextUnformatted(text);
}

void draw_shortcut_row(const char *title, const char *key) {
  const float center = ImGui::GetContentRegionAvail().x * 0.5f;
  const float title_w = ImGui::CalcTextSize(title).x;
  ImGui::SetCursorPosX(center - title_w - ImGui::GetStyle().ItemSpacing.x);
  ImGui::AlignTextToFramePadding();
  ImGui::TextDisabled("%s", title);
  ImGui::SameLine(center);
  ImGui::BeginDisabled();
  ImGui::Button(key);
  ImGui::EndDisabled();
}

// mirrors CenterWidget::createWelcomeWidget() in tools/cabana/detailwidget.cc
void draw_welcome(const ImVec2 &pos, const ImVec2 &size) {
  ImGui::SetNextWindowPos(pos);
  ImGui::SetNextWindowSize(size);
  ImGui::PushStyleColor(ImGuiCol_WindowBg, ImGui::GetStyleColorVec4(ImGuiCol_ChildBg));
  const ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                                 ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNav;
  if (ImGui::Begin("##welcome", nullptr, flags)) {
    const float content_height = 50.0f + 4 * (ImGui::GetFrameHeight() + ImGui::GetStyle().ItemSpacing.y);
    ImGui::SetCursorPosY((size.y - content_height) * 0.5f);
    push_bold_font(50.0f);
    draw_centered_text("CABANA");
    pop_bold_font();
    ImGui::Spacing();
    ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
    draw_centered_text("<- Select a message to view details");
    ImGui::PopStyleColor();
    ImGui::Spacing();
    draw_shortcut_row("Pause", "Space");
    draw_shortcut_row("Help", "F1");
    draw_shortcut_row("WhatsThis", "Shift+F1");
  }
  ImGui::End();
  ImGui::PopStyleColor();
}

void draw_status_bar(AppState *state, const ImVec2 &pos, float width) {
  ImGui::SetNextWindowPos(pos);
  ImGui::SetNextWindowSize(ImVec2(width, STATUS_BAR_HEIGHT));
  const ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                                 ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNav;
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8.0f, 5.0f));
  if (ImGui::Begin("##statusbar", nullptr, flags)) {
    ImGui::TextUnformatted(state->status_text.c_str());
  }
  ImGui::End();
  ImGui::PopStyleVar();
}

void draw_about_popup(AppState *state) {
  if (state->show_about) {
    ImGui::OpenPopup("About Cabana");
    state->show_about = false;
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

void draw_ui(AppState *state) {
  const ImGuiIO &io = ImGui::GetIO();
  const bool ctrl = io.KeyCtrl || io.KeySuper;
  if (!io.WantTextInput && ctrl && ImGui::IsKeyPressed(ImGuiKey_Q, false)) {
    state->request_close = true;
  }

  draw_menu_bar(state);

  const ImGuiViewport *viewport = ImGui::GetMainViewport();
  const ImVec2 work_pos = viewport->WorkPos;
  const ImVec2 work_size = viewport->WorkSize;
  const float content_height = work_size.y - STATUS_BAR_HEIGHT;
  draw_welcome(work_pos, ImVec2(work_size.x, content_height));
  draw_status_bar(state, ImVec2(work_pos.x, work_pos.y + content_height), work_size.x);
  draw_about_popup(state);
}

void render_frame(GLFWwindow *window, AppState *state, const fs::path *capture_path) {
  glfwPollEvents();

  if (state->theme_changed) {
    apply_theme(state->theme);
    state->theme_changed = false;
  }

  int framebuffer_width = 0;
  int framebuffer_height = 0;
  glfwGetFramebufferSize(window, &framebuffer_width, &framebuffer_height);

  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  draw_ui(state);

  ImGui::Render();
  if (state->request_close) {
    glfwSetWindowShouldClose(window, GLFW_TRUE);
    state->request_close = false;
  }

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

int run(const Options &options) {
  GlfwRuntime glfw_runtime(options);
  ImGuiRuntime imgui_runtime(glfw_runtime.window());
  load_fonts();

  AppState state;
  state.theme = options.dark_theme ? Theme::Dark : Theme::Light;
  apply_theme(state.theme);

  if (!options.output_path.empty()) {
    const fs::path capture_path = options.output_path;
    for (int i = 0; i < 3; ++i) {
      render_frame(glfw_runtime.window(), &state, nullptr);
    }
    render_frame(glfw_runtime.window(), &state, &capture_path);
    if (!options.show) return 0;
  }

  while (!glfwWindowShouldClose(glfw_runtime.window())) {
    render_frame(glfw_runtime.window(), &state, nullptr);
  }
  return 0;
}
