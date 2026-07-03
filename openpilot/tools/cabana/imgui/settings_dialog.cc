// Settings dialog -- ImGui port of the frozen Qt SettingsDlg
// (tools/cabana/settings.cc, recovered from git history at
// `git show c547be3f71~1:openpilot/tools/cabana/settings.cc` since that class
// was deleted when the Qt-free core landed). Same three groups, same labels,
// same ranges/steps, same live-stream-logging checkable group with a
// directory browse button. OK applies + persists to disk immediately
// (settings.save()) and fires settings.changed(), matching SettingsDlg::save().
// Cancel discards: the dialog only ever writes into its own snapshot fields
// until OK is pressed, so simply closing without applying is the "restore".

#include "tools/cabana/imgui/app.h"

#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <system_error>

#include "imgui_internal.h"  // ImGui::FindWindowByName, ImGuiWindow::Active (see draw_settings_dialog)

#include "tools/cabana/imgui/file_dialog.h"
#include "tools/cabana/settings.h"

namespace {

namespace fs = std::filesystem;

// tools/cabana/settings.cc: `const int MIN_CACHE_MINIUTES = 30;` / `MAX_CACHE_MINIUTES = 120;`
constexpr int MIN_CACHE_MINUTES = 30;
constexpr int MAX_CACHE_MINUTES = 120;
constexpr int MIN_FPS = 10;
constexpr int MAX_FPS = 100;
constexpr int MIN_CHART_HEIGHT = 100;
constexpr int MAX_CHART_HEIGHT = 500;

// SettingsDlg ctor: theme->addItems({tr("Automatic"), tr("Light"), tr("Dark")});
constexpr const char *kThemeItems[] = {"Automatic", "Light", "Dark"};
// SettingsDlg ctor: drag_direction->addItems({"MSB First", "LSB First", "Always Little Endian", "Always Big Endian"});
constexpr const char *kDragDirectionItems[] = {"MSB First", "LSB First", "Always Little Endian", "Always Big Endian"};

struct SettingsDialogState {
  bool need_open = false;
  bool active = false;
  bool reopen_after_browse = false;  // true while the log-path file dialog is up (see draw_settings_dialog)

  int theme_idx = 0;
  int fps = 10;
  int cached_minutes = 30;
  int chart_height = 200;
  int drag_direction_idx = 0;
  bool log_livestream = true;
  char log_path[1024] = {};
};

SettingsDialogState g_dlg;

std::string home_dir() {
  const char *home = std::getenv("HOME");
  return home != nullptr ? std::string(home) : std::string();
}

// Snapshot the live settings into the dialog's own fields on open -- this
// snapshot doubles as Cancel's "restore" since `settings` itself is only
// ever touched from apply_and_save() below (mirrors opening a fresh
// SettingsDlg each time in Qt, which always read straight from `settings`).
void snapshot_from_settings() {
  g_dlg.theme_idx = std::clamp(settings.theme, 0, 2);
  g_dlg.fps = settings.fps;
  g_dlg.cached_minutes = settings.max_cached_minutes;
  g_dlg.chart_height = settings.chart_height;
  g_dlg.drag_direction_idx = std::clamp(static_cast<int>(settings.drag_direction), 0, 3);
  g_dlg.log_livestream = settings.log_livestream;
  std::snprintf(g_dlg.log_path, sizeof(g_dlg.log_path), "%s", settings.log_path.c_str());
}

// mirrors SignalView-style property-form rows (see signal_view.cc
// begin_field_row): label left-aligned, control starting at a fixed column.
void begin_field_row(const char *label, float col2_x) {
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted(label);
  ImGui::SameLine(col2_x);
}

// mirrors SettingsDlg::save()
void apply_and_save(AppState &app) {
  if (settings.theme != g_dlg.theme_idx) {
    settings.theme = g_dlg.theme_idx;
    // Live-apply, same as the View menu's "Dark Theme" toggle in app.cc:
    // apply_theme() re-skins the running ImGui style on the very next
    // frame, so (unlike Qt, whose tooltip warned "you may need to restart
    // cabana after changing theme") this port needs no restart note here.
    app.theme = (settings.theme == DARK_THEME) ? Theme::Dark : Theme::Light;
    app.theme_changed = true;
  }
  settings.fps = g_dlg.fps;
  settings.max_cached_minutes = g_dlg.cached_minutes;
  settings.chart_height = g_dlg.chart_height;
  settings.log_livestream = g_dlg.log_livestream;
  settings.log_path = g_dlg.log_path;
  settings.drag_direction = static_cast<Settings::DragDirection>(g_dlg.drag_direction_idx);
  settings.save();
  settings.changed();
}

}  // namespace

void open_settings_dialog() {
  snapshot_from_settings();
  g_dlg.need_open = true;
  g_dlg.active = true;
}

void draw_settings_dialog(AppState &app) {
  constexpr const char *kPopupId = "Settings##cabana_settings";
  // file_dialog.cc's file_dialog_draw(): `constexpr const char *kPopupId = "File Browser##dbc_file_dialog";`
  constexpr const char *kFileDialogWindowTitle = "File Browser##dbc_file_dialog";

  if (g_dlg.reopen_after_browse) {
    // The log-path Browse button (below) hands off to file_dialog.cc's
    // modal. file_dialog_draw() is invoked unconditionally from
    // dbc_menus.cc's draw_dbc_modals(), which runs *earlier* in draw_ui()'s
    // per-frame call order than draw_settings_dialog() -- so its
    // BeginPopupModal isn't submitted as an ImGui child of ours, it's just
    // another modal opened the same frame, and ImGui's popup stack doesn't
    // let two unrelated modals stay open at once (accepting the file
    // dialog silently pops ours off the stack too). So instead of trying to
    // keep our popup open underneath it, we close ours outright before
    // opening the file dialog (Browse handler below) and poll for *its*
    // window here to reopen ours the moment it goes away.
    //
    // FindWindowByName() is checked *after* file_dialog_draw() already ran
    // this frame (see draw_ui() ordering), so ->Active -- set the instant
    // Begin() runs, false again from NewFrame() until Begin() runs -- tells
    // us with zero lag whether it's still up this frame; ->WasActive would
    // read one frame stale here. g_dlg's fields (including the just-picked
    // log_path) are left untouched since this path bypasses
    // snapshot_from_settings().
    const ImGuiWindow *fd_win = ImGui::FindWindowByName(kFileDialogWindowTitle);
    if (fd_win == nullptr || !fd_win->Active) {
      g_dlg.reopen_after_browse = false;
      g_dlg.need_open = true;
      g_dlg.active = true;
    }
  }

  if (g_dlg.need_open) {
    ImGui::OpenPopup(kPopupId);
    g_dlg.need_open = false;
  }
  if (!g_dlg.active) return;

  ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
  // Qt: setFixedSize(400, sizeHint().height()) -- fixed width, auto height.
  ImGui::SetNextWindowSizeConstraints(ImVec2(400.0f, 0.0f), ImVec2(400.0f, FLT_MAX));
  if (!ImGui::BeginPopupModal(kPopupId, nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings)) {
    return;
  }

  // Column where every control starts -- sized to the longest label in this
  // dialog, mirroring QFormLayout's auto-sized label column.
  const float col2 = ImGui::CalcTextSize("Max Cached Minutes").x + ImGui::GetStyle().ItemSpacing.x + 8.0f;

  // -- General (QGroupBox "General") ---------------------------------------
  ImGui::SeparatorText("General");

  begin_field_row("Color Theme", col2);
  ImGui::SetNextItemWidth(-FLT_MIN);
  if (ImGui::BeginCombo("##theme", kThemeItems[g_dlg.theme_idx])) {
    for (int i = 0; i < 3; ++i) {
      if (ImGui::Selectable(kThemeItems[i], i == g_dlg.theme_idx)) g_dlg.theme_idx = i;
    }
    ImGui::EndCombo();
  }

  begin_field_row("FPS", col2);
  ImGui::SetNextItemWidth(-FLT_MIN);
  ImGui::InputInt("##fps", &g_dlg.fps, 10, 10);
  g_dlg.fps = std::clamp(g_dlg.fps, MIN_FPS, MAX_FPS);
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Range %d-%d, step 10", MIN_FPS, MAX_FPS);

  begin_field_row("Max Cached Minutes", col2);
  ImGui::SetNextItemWidth(-FLT_MIN);
  ImGui::InputInt("##cached_minutes", &g_dlg.cached_minutes, 1, 10);
  g_dlg.cached_minutes = std::clamp(g_dlg.cached_minutes, MIN_CACHE_MINUTES, MAX_CACHE_MINUTES);

  // -- New Signal Settings (QGroupBox "New Signal Settings") ---------------
  ImGui::SeparatorText("New Signal Settings");

  begin_field_row("Drag Direction", col2);
  ImGui::SetNextItemWidth(-FLT_MIN);
  if (ImGui::BeginCombo("##drag_direction", kDragDirectionItems[g_dlg.drag_direction_idx])) {
    for (int i = 0; i < 4; ++i) {
      if (ImGui::Selectable(kDragDirectionItems[i], i == g_dlg.drag_direction_idx)) g_dlg.drag_direction_idx = i;
    }
    ImGui::EndCombo();
  }

  // -- Chart (QGroupBox "Chart") --------------------------------------------
  ImGui::SeparatorText("Chart");

  begin_field_row("Chart Height", col2);
  ImGui::SetNextItemWidth(-FLT_MIN);
  ImGui::InputInt("##chart_height", &g_dlg.chart_height, 10, 10);
  g_dlg.chart_height = std::clamp(g_dlg.chart_height, MIN_CHART_HEIGHT, MAX_CHART_HEIGHT);

  // -- live-stream logging (checkable QGroupBox "Enable live stream logging") --
  ImGui::SeparatorText("Live Stream Logging");
  ImGui::Checkbox("Enable live stream logging", &g_dlg.log_livestream);
  ImGui::BeginDisabled(!g_dlg.log_livestream);
  ImGui::SetNextItemWidth(std::max(80.0f, ImGui::GetContentRegionAvail().x - 92.0f));
  ImGui::InputText("##log_path", g_dlg.log_path, sizeof(g_dlg.log_path), ImGuiInputTextFlags_ReadOnly);
  ImGui::SameLine();
  // Qt: QFileDialog::getExistingDirectory(this, tr("Log File Location"),
  //     QStandardPaths::writableLocation(QStandardPaths::HomeLocation), ShowDirsOnly)
  if (ImGui::Button("Browse...", ImVec2(80.0f, 0.0f))) {
    // Close our own modal first -- see the reopen_after_browse comment atop
    // this function for why two independently-opened modals can't coexist.
    g_dlg.reopen_after_browse = true;
    g_dlg.active = false;
    ImGui::CloseCurrentPopup();
    file_dialog_open(FileDialogMode::Save, "Log File Location", home_dir(), "", ".", [](const std::string &path) {
      // file_dialog.cc joins current_dir/filename; default_filename "."
      // selects "this directory" itself -- collapse the resulting
      // ".../<dir>/." back down to a clean directory path.
      std::error_code ec;
      const fs::path canon = fs::weakly_canonical(path, ec);
      const std::string chosen = (!ec ? canon : fs::path(path)).string();
      std::snprintf(g_dlg.log_path, sizeof(g_dlg.log_path), "%s", chosen.c_str());
    });
  }
  ImGui::EndDisabled();

  ImGui::Spacing();
  ImGui::Separator();
  ImGui::Spacing();

  // QDialogButtonBox(Ok | Cancel)
  if (ImGui::Button("OK", ImVec2(100.0f, 0.0f))) {
    apply_and_save(app);
    g_dlg.active = false;
    ImGui::CloseCurrentPopup();
  }
  ImGui::SameLine();
  if (ImGui::Button("Cancel", ImVec2(100.0f, 0.0f)) || ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
    // Discard: `settings` was never touched, only g_dlg's snapshot copy.
    g_dlg.active = false;
    ImGui::CloseCurrentPopup();
  }

  ImGui::EndPopup();
}
