#include "tools/cabana/ui/dialogs/settingsdialog.h"

#include <algorithm>
#include <utility>

#include "imgui.h"
#include "imgui_internal.h"
#include "tools/cabana/settings.h"
#include "tools/cabana/ui/dialogs/filedialog.h"
#include "tools/cabana/ui/util.h"
#include "tools/cabana/utils/util.h"

const int MIN_CACHE_MINIUTES = 30;
const int MAX_CACHE_MINIUTES = 120;

namespace {

// the label sits in the left column, the field in the right one, all fields aligned
const char *kFormLabels[] = {"Color Theme", "Max Cached Minutes", "Drag Direction", "Chart Height"};

float formLabelWidth() {
  float w = 0.0f;
  for (const char *label : kFormLabels) w = std::max(w, ImGui::CalcTextSize(label).x);
  return w + ImGui::GetStyle().ItemSpacing.x * 2;  // horizontal spacing between label and field
}

void formRow(const char *label, float label_width) {
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted(label);
  ImGui::SameLine(label_width);
  ImGui::SetNextItemWidth(-FLT_MIN);
}

}  // namespace

void SettingsDialog::open() {
  theme_ = settings.theme;
  cached_minutes_ = settings.max_cached_minutes;
  drag_direction_ = settings.drag_direction;
  chart_height_ = settings.chart_height;
  log_livestream_ = settings.log_livestream;
  log_path_ = settings.log_path;
  open_ = true;
  popup_.reset();
}

void SettingsDialog::draw() {
  if (!open_) return;
  if (!beginDialog("Settings", &popup_, ImVec2(400.0f, 0.0f))) return;
  const float label_width = formLabelWidth();

  ImGui::SeparatorText("General");
  static const char *themes[] = {"Automatic", "Light", "Dark"};
  formRow("Color Theme", label_width);
  ImGui::Combo("##Color Theme", &theme_, themes, 3);
  ImGui::SetItemTooltip("You may need to restart cabana after changes theme");
  formRow("Max Cached Minutes", label_width);
  // InputInt takes no character filter, so out of range text is clamped after the edit
  if (ImGui::InputInt("##Max Cached Minutes", &cached_minutes_, 1, 10)) {
    cached_minutes_ = std::clamp(cached_minutes_, MIN_CACHE_MINIUTES, MAX_CACHE_MINIUTES);
  }

  ImGui::SeparatorText("New Signal Settings");
  static const char *directions[] = {"MSB First", "LSB First", "Always Little Endian", "Always Big Endian"};
  formRow("Drag Direction", label_width);
  ImGui::Combo("##Drag Direction", &drag_direction_, directions, 4);

  ImGui::SeparatorText("Chart");
  formRow("Chart Height", label_width);
  if (ImGui::InputInt("##Chart Height", &chart_height_, 10, 10)) chart_height_ = std::clamp(chart_height_, 100, 500);

  checkBox("Enable live stream logging", &log_livestream_);
  ImGui::BeginDisabled(!log_livestream_);
  ImGui::SetNextItemWidth(-90.0f);
  inputText("##log_path", &log_path_, "", ImGuiInputTextFlags_ReadOnly);
  ImGui::SameLine();
  if (ImGui::Button("Browse...")) {
    FileDialog::getExistingDirectory("Log File Location", utils::homePath(), [this](const std::string &fn) {
      if (!fn.empty()) log_path_ = fn;
    });
  }
  ImGui::EndDisabled();

  ImGui::Separator();
  bool accepted = false, done = false;
  dialogButtons("OK", &accepted, &done);
  if (accepted) {
    save();
    done = true;
  }
  if (dialogEscapePressed()) done = true;
  FileDialog::draw();  // nested so the directory picker stacks on this modal
  if (done) {
    open_ = false;
    ImGui::CloseCurrentPopup();
  }
  ImGui::EndPopup();
}

void SettingsDialog::save() {
  if (std::exchange(settings.theme, theme_) != settings.theme) {
    // set the theme before notifying
    applyTheme(settings.theme);
  }
  settings.max_cached_minutes = cached_minutes_;
  settings.chart_height = chart_height_;
  settings.log_livestream = log_livestream_;
  settings.log_path = log_path_;
  settings.drag_direction = (Settings::DragDirection)drag_direction_;
  settings.changed();
}
