#include "tools/cabana/ui/dialogs/settingsdialog.h"

#include <utility>

#include "imgui.h"
#include "imgui_internal.h"
#include "tools/cabana/settings.h"
#include "tools/cabana/ui/app.h"
#include "tools/cabana/ui/dialogs/filedialog.h"
#include "tools/cabana/utils/util.h"

const int MIN_CACHE_MINIUTES = 30;
const int MAX_CACHE_MINIUTES = 120;

void SettingsDialog::open() {
  theme_ = settings.theme;
  fps_ = settings.fps;
  cached_minutes_ = settings.max_cached_minutes;
  drag_direction_ = settings.drag_direction;
  chart_height_ = settings.chart_height;
  log_livestream_ = settings.log_livestream;
  log_path_ = settings.log_path;
  open_ = true;
  show_ = false;
}

void SettingsDialog::draw() {
  if (!open_) return;
  if (!show_) {
    ImGui::OpenPopup("Settings");
    show_ = true;
  }
  ImGui::SetNextWindowSize(ImVec2(400.0f, 0.0f), ImGuiCond_Appearing);
  ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
  if (!ImGui::BeginPopupModal("Settings", nullptr, ImGuiWindowFlags_NoResize)) return;
  const ImGuiInputTextFlags spin_flags = 0;  // InputInt forbids EnterReturnsTrue; it applies the text on every edit

  ImGui::SeparatorText("General");
  static const char *themes[] = {"Automatic", "Light", "Dark"};
  ImGui::Combo("Color Theme", &theme_, themes, 3);
  ImGui::SetItemTooltip("You may need to restart cabana after changes theme");
  if (ImGui::InputInt("FPS", &fps_, 10, 10, spin_flags)) fps_ = std::clamp(fps_, 10, 100);
  if (ImGui::InputInt("Max Cached Minutes", &cached_minutes_, 1, 10, spin_flags)) {
    cached_minutes_ = std::clamp(cached_minutes_, MIN_CACHE_MINIUTES, MAX_CACHE_MINIUTES);
  }

  ImGui::SeparatorText("New Signal Settings");
  static const char *directions[] = {"MSB First", "LSB First", "Always Little Endian", "Always Big Endian"};
  ImGui::Combo("Drag Direction", &drag_direction_, directions, 4);

  ImGui::SeparatorText("Chart");
  if (ImGui::InputInt("Chart Height", &chart_height_, 10, 10, spin_flags)) chart_height_ = std::clamp(chart_height_, 100, 500);

  ImGui::Checkbox("Enable live stream logging", &log_livestream_);
  ImGui::BeginDisabled(!log_livestream_);
  ImGui::SetNextItemWidth(-90.0f);
  ImGui::InputText("##log_path", log_path_.data(), log_path_.size() + 1, ImGuiInputTextFlags_ReadOnly);
  ImGui::SameLine();
  if (ImGui::Button("Browse...")) {
    FileDialog::getExistingDirectory("Log File Location", utils::homePath(), [this](const std::string &fn) {
      if (!fn.empty()) log_path_ = fn;
    });
  }
  ImGui::EndDisabled();

  ImGui::Separator();
  bool done = false;
  if (ImGui::Button("OK", ImVec2(80.0f, 0.0f))) {
    save();
    done = true;
  }
  ImGui::SameLine();
  if (ImGui::Button("Cancel", ImVec2(80.0f, 0.0f))) done = true;
  if (ImGui::GetTopMostPopupModal() == ImGui::GetCurrentWindow() && ImGui::IsKeyPressed(ImGuiKey_Escape, false)) done = true;
  FileDialog::draw();  // nested so the directory picker stacks on this modal
  if (done) {
    open_ = false;
    ImGui::CloseCurrentPopup();
  }
  ImGui::EndPopup();
}

void SettingsDialog::save() {
  if (std::exchange(settings.theme, theme_) != settings.theme) {
    // set theme before emit changed
    applyTheme(settings.theme);
  }
  settings.fps = fps_;
  settings.max_cached_minutes = cached_minutes_;
  settings.chart_height = chart_height_;
  settings.log_livestream = log_livestream_;
  settings.log_path = log_path_;
  settings.drag_direction = (Settings::DragDirection)drag_direction_;
  settings.changed();
}
