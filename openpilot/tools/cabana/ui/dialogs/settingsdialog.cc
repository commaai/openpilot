#include "tools/cabana/ui/dialogs/settingsdialog.h"

#include <algorithm>
#include <utility>

#include "imgui.h"
#include "imgui_internal.h"
#include "tools/cabana/settings.h"
#include "tools/cabana/ui/dialogs/filedialog.h"
#include "tools/cabana/ui/util.h"
#include "tools/cabana/utils/util.h"

namespace {

const int MIN_CACHE_MINUTES = 30;
const int MAX_CACHE_MINUTES = 120;

// the label sits in the left column, the field in the right one, all fields aligned
enum FormLabel { THEME, CACHED_MINUTES, DRAG_DIRECTION, CHART_HEIGHT, FORM_LABEL_COUNT };
const char *FORM_LABELS[FORM_LABEL_COUNT] = {"Color Theme", "Max Cached Minutes", "Drag Direction", "Chart Height"};

float formLabelWidth() {
  float w = 0.0f;
  for (const char *label : FORM_LABELS) w = std::max(w, ImGui::CalcTextSize(label).x);
  return w + ImGui::GetStyle().ItemSpacing.x * 2;  // horizontal spacing between label and field
}

void formRow(FormLabel label, float label_width) {
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted(FORM_LABELS[label]);
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
  static const char *themes[] = {"Light", "Dark"};
  formRow(THEME, label_width);
  int theme_index = theme_ - LIGHT_THEME;
  if (ImGui::Combo("##theme", &theme_index, themes, IM_ARRAYSIZE(themes))) theme_ = theme_index + LIGHT_THEME;
  formRow(CACHED_MINUTES, label_width);
  // InputInt takes no character filter, so out of range text is clamped after the edit
  if (ImGui::InputInt("##cached_minutes", &cached_minutes_, 1, 10)) {
    cached_minutes_ = std::clamp(cached_minutes_, MIN_CACHE_MINUTES, MAX_CACHE_MINUTES);
  }

  ImGui::SeparatorText("New Signal Settings");
  static const char *directions[] = {"MSB First", "LSB First", "Always Little Endian", "Always Big Endian"};
  formRow(DRAG_DIRECTION, label_width);
  ImGui::Combo("##drag_direction", &drag_direction_, directions, IM_ARRAYSIZE(directions));

  ImGui::SeparatorText("Chart");
  formRow(CHART_HEIGHT, label_width);
  if (ImGui::InputInt("##chart_height", &chart_height_, 10, 10)) chart_height_ = std::clamp(chart_height_, 100, 500);

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
  FileDialog::draw();  // nested so the directory picker stacks on this modal
  if (done) {
    open_ = false;
    ImGui::CloseCurrentPopup();
  }
  ImGui::EndPopup();
}

void SettingsDialog::save() {
  if (std::exchange(settings.theme, theme_) != settings.theme) applyTheme(settings.theme);
  settings.max_cached_minutes = cached_minutes_;
  settings.chart_height = chart_height_;
  settings.log_livestream = log_livestream_;
  settings.log_path = log_path_;
  settings.drag_direction = (Settings::DragDirection)drag_direction_;
  settings.changed();
}
