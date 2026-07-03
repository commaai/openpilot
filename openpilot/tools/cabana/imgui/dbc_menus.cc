#include "tools/cabana/imgui/dbc_menus.h"

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "imgui.h"

#include <GLFW/glfw3.h>

#include "json11/json11.hpp"

#include "tools/cabana/commands.h"
#include "tools/cabana/dbc/dbcmanager.h"
#include "tools/cabana/imgui/app.h"
#include "tools/cabana/imgui/file_dialog.h"
#include "tools/cabana/settings.h"
#include "tools/cabana/streams/abstractstream.h"

// File/Edit menu items, open/save/clipboard flows, recent files, opendbc
// loading, fingerprint auto-load, window title -- parity spec:
// tools/cabana/mainwin.{h,cc} (frozen Qt reference), specifically
// createActions(), loadFile/newFile/openFile/save/saveAs/saveToClipboard/
// loadFromClipboard/closeFile, remindSaveChanges, updateRecentFiles/
// updateRecentFileMenu, loadFingerprints/eventsMerged, updateLoadSaveMenus,
// undoStackCleanChanged/DBCFileChanged.

namespace fs = std::filesystem;

namespace {

constexpr int MAX_RECENT_FILES = 15;

GLFWwindow *g_window = nullptr;

// -- window title -----------------------------------------------------------

bool g_title_dirty = true;
void mark_title_dirty() { g_title_dirty = true; }

void refresh_window_title_if_dirty() {
  if (!g_title_dirty || g_window == nullptr) return;
  g_title_dirty = false;

  std::string title;
  for (DBCFile *f : dbc()->allDBCFiles()) {
    if (!title.empty()) title += " | ";
    title += "(" + toString(dbc()->sources(f)) + ") " + f->name();
  }
  if (title.empty()) {
    title = "Cabana";
  } else {
    if (!UndoStack::instance()->isClean()) title += "*";
    title += " - Cabana";
  }
  glfwSetWindowTitle(g_window, title.c_str());
}

// -- simple error/info modal (mirrors the QMessageBox::warning/information
// call sites in mainwin.cc) --------------------------------------------------

struct SimpleModalState {
  bool need_open = false;
  bool active = false;
  bool is_error = false;
  std::string title, message, detail;
};
SimpleModalState g_simple;

void open_error_modal(const std::string &title, const std::string &message, const std::string &detail) {
  g_simple = {true, true, true, title, message, detail};
}
void open_info_modal(const std::string &title, const std::string &message) {
  g_simple = {true, true, false, title, message, std::string()};
}

// Single fixed popup ID shared by both error and info variants (only one can
// ever be pending at a time in this UI's flows) so the ImGui window identity
// never depends on caller-supplied text.
void draw_simple_modal() {
  constexpr const char *kPopupId = "Cabana##dbc_notice";

  if (g_simple.need_open) {
    ImGui::OpenPopup(kPopupId);
    g_simple.need_open = false;
  }
  if (!g_simple.active) return;

  ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
  ImGui::SetNextWindowSizeConstraints(ImVec2(280.0f, 0.0f), ImVec2(520.0f, FLT_MAX));
  if (ImGui::BeginPopupModal(kPopupId, nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    if (g_simple.is_error) {
      ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.85f, 0.35f, 0.3f, 1.0f));
    }
    push_bold_font();
    ImGui::TextUnformatted(g_simple.title.c_str());
    pop_bold_font();
    if (g_simple.is_error) ImGui::PopStyleColor();
    ImGui::Spacing();
    ImGui::TextWrapped("%s", g_simple.message.c_str());
    if (!g_simple.detail.empty()) {
      ImGui::Spacing();
      ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
      ImGui::TextWrapped("%s", g_simple.detail.c_str());
      ImGui::PopStyleColor();
    }
    ImGui::Spacing();
    if (ImGui::Button("OK", ImVec2(100.0f, 0.0f)) || ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
      ImGui::CloseCurrentPopup();
      g_simple.active = false;
    }
    ImGui::EndPopup();
  }
}

// -- unsaved-changes reminder (mirrors MainWindow::remindSaveChanges) -------

struct RemindState {
  bool active = false;
  bool need_open = false;
  std::function<void()> on_resolved;
};
RemindState g_remind;

void ui_save_all(std::function<void()> done);

// UndoStack::instance()->isClean() true -> resolves synchronously (matches
// Qt's while(!isClean()) never entering the loop, then the unconditional
// clear() after it). Otherwise opens the "Unsaved Changes" modal; if another
// remind flow is already in progress, the new request is dropped (Qt's
// QMessageBox::exec() is blocking, so only one can ever be in flight there).
void remind_save_changes(std::function<void()> on_resolved) {
  if (UndoStack::instance()->isClean()) {
    UndoStack::instance()->clear();
    if (on_resolved) on_resolved();
    return;
  }
  if (g_remind.active) return;
  g_remind.active = true;
  g_remind.need_open = true;
  g_remind.on_resolved = std::move(on_resolved);
}

void draw_remind_modal() {
  constexpr const char *kPopupId = "Unsaved Changes";

  if (g_remind.need_open) {
    ImGui::OpenPopup(kPopupId);
    g_remind.need_open = false;
  }
  if (!g_remind.active) return;

  ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
  if (ImGui::BeginPopupModal(kPopupId, nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::TextWrapped("You have unsaved changes. Press ok to save them, cancel to discard.");
    ImGui::Spacing();
    if (ImGui::Button("OK", ImVec2(100.0f, 0.0f))) {
      ImGui::CloseCurrentPopup();
      ui_save_all([]() {
        if (!UndoStack::instance()->isClean()) {
          // A Save As in the chain was cancelled -- ask again (mirrors the
          // Qt while loop re-checking isClean() before looping back).
          g_remind.need_open = true;
        } else {
          UndoStack::instance()->clear();
          g_remind.active = false;
          if (g_remind.on_resolved) g_remind.on_resolved();
          g_remind.on_resolved = nullptr;
        }
      });
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel", ImVec2(100.0f, 0.0f)) || ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
      ImGui::CloseCurrentPopup();
      UndoStack::instance()->clear();
      g_remind.active = false;
      if (g_remind.on_resolved) g_remind.on_resolved();
      g_remind.on_resolved = nullptr;
    }
    ImGui::EndPopup();
  }
}

// -- recent files (mirrors updateRecentFiles/updateRecentFileMenu) ----------

void note_recent_file(const std::string &fn) {
  auto &rf = settings.recent_files;
  rf.erase(std::remove(rf.begin(), rf.end(), fn), rf.end());
  rf.insert(rf.begin(), fn);
  while (static_cast<int>(rf.size()) > MAX_RECENT_FILES) rf.pop_back();

  std::error_code ec;
  fs::path abs = fs::absolute(fs::path(fn), ec);
  settings.last_dir = (!ec ? abs : fs::path(fn)).parent_path().string();
}

// -- core open/save/close flows (mirrors newFile/openFile/loadFile/
// loadFromClipboard/save/saveAs/saveToClipboard/closeFile) ------------------

void save_file_as(DBCFile *f, std::function<void()> on_done) {
  const std::string title = "Save File (bus: " + toString(dbc()->sources(f)) + ")";
  file_dialog_open(FileDialogMode::Save, title, settings.last_dir, ".dbc", "untitled.dbc",
                    [f, on_done](const std::string &path) {
                      f->saveAs(path);
                      UndoStack::instance()->setClean();
                      note_recent_file(path);
                      mark_title_dirty();
                      if (on_done) on_done();
                    });
}

void ui_save_queue(std::vector<DBCFile *> files, size_t idx, std::function<void()> done, bool force_as) {
  if (idx >= files.size()) {
    if (done) done();
    return;
  }
  DBCFile *f = files[idx];
  auto next = [files, idx, done, force_as]() { ui_save_queue(files, idx + 1, done, force_as); };
  if (!force_as && !f->filename.empty()) {
    f->save();
    UndoStack::instance()->setClean();
    next();
  } else if (!f->isEmpty()) {
    save_file_as(f, next);
  } else {
    next();
  }
}

// mirrors MainWindow::save(): saves every non-empty open DBC file, prompting
// Save As for any that don't have a filename yet.
void ui_save_all(std::function<void()> done = nullptr) {
  std::vector<DBCFile *> files;
  for (DBCFile *f : dbc()->allDBCFiles()) {
    if (!f->isEmpty()) files.push_back(f);
  }
  ui_save_queue(std::move(files), 0, std::move(done), /*force_as=*/false);
}

// mirrors MainWindow::saveAs(): always prompts, even if a filename exists.
void ui_save_as_all() {
  std::vector<DBCFile *> files;
  for (DBCFile *f : dbc()->allDBCFiles()) {
    if (!f->isEmpty()) files.push_back(f);
  }
  ui_save_queue(std::move(files), 0, nullptr, /*force_as=*/true);
}

void ui_save_file(DBCFile *f) {
  if (!f->filename.empty()) {
    f->save();
    UndoStack::instance()->setClean();
  } else if (!f->isEmpty()) {
    save_file_as(f, nullptr);
  }
}

// per-file "Copy to Clipboard..." (Manage DBC Files submenu)
void ui_copy_to_clipboard(DBCFile *f) {
  if (f->isEmpty() || g_window == nullptr) return;
  glfwSetClipboardString(g_window, f->generateDBC().c_str());
  open_info_modal("Copy To Clipboard", "DBC Successfully copied!");
}

// top-level "Copy DBC To Clipboard": mirrors saveToClipboard(), which the Qt
// source itself flags as "Should not be called with more than 1 file open" --
// the first non-empty file wins here too (see report for the accepted
// multi-bus-clipboard limitation).
void ui_copy_to_clipboard_all() {
  for (DBCFile *f : dbc()->allDBCFiles()) {
    if (f->isEmpty()) continue;
    ui_copy_to_clipboard(f);
    break;
  }
}

void ui_close_file_sources(SourceSet s, std::function<void()> then) {
  remind_save_changes([s, then]() {
    if (s == SOURCE_ALL) dbc()->closeAll(); else dbc()->close(s);
    if (then) then();
  });
}

void ui_close_dbc_file(DBCFile *f) {
  remind_save_changes([f]() {
    dbc()->close(f);
    if (dbc()->dbcCount() == 0) {
      // Ensure we always have at least one file open (mirrors closeFile(DBCFile*)'s newFile() fallback).
      dbc()->open(SOURCE_ALL, std::string(), std::string());
    }
  });
}

void ui_new_file(SourceSet s) {
  ui_close_file_sources(s, [s]() { dbc()->open(s, std::string(), std::string()); });
}

void ui_load_file(const std::string &fn, SourceSet s) {
  if (fn.empty()) return;
  ui_close_file_sources(s, [fn, s]() {
    std::string error;
    if (dbc()->open(s, fn, &error)) {
      note_recent_file(fn);
    } else {
      open_error_modal("Failed to load DBC file", "Failed to parse DBC file " + fn, error);
    }
  });
}

void ui_open_file(SourceSet s) {
  remind_save_changes([s]() {
    file_dialog_open(FileDialogMode::Open, "Open File", settings.last_dir, ".dbc", std::string(),
                      [s](const std::string &path) { ui_load_file(path, s); });
  });
}

void ui_load_from_clipboard(SourceSet s) {
  ui_close_file_sources(s, [s]() {
    const char *clip = g_window != nullptr ? glfwGetClipboardString(g_window) : nullptr;
    const std::string text = clip != nullptr ? clip : "";
    std::string error;
    const bool ok = dbc()->open(s, std::string(), text, &error);
    if (ok && dbc()->nonEmptyDBCCount() > 0) {
      open_info_modal("Load From Clipboard", "DBC Successfully Loaded!");
    } else {
      open_error_modal("Failed to load DBC from clipboard", "Make sure that you paste the text with correct format.", error);
    }
  });
}

void ui_load_dbc_from_opendbc(const std::string &name) {
  ui_load_file(std::string(OPENDBC_FILE_PATH) + "/" + name, SOURCE_ALL);
}

// -- fingerprint auto-load (mirrors loadFingerprints() + eventsMerged()) ----

std::map<std::string, std::string> g_fingerprint_to_dbc;
std::string g_last_car_fingerprint;

void load_fingerprints() {
  const fs::path json_path = repo_root() / "openpilot" / "tools" / "cabana" / "dbc" / "car_fingerprint_to_dbc.json";
  std::ifstream file(json_path, std::ios::binary);
  if (!file.is_open()) return;
  std::ostringstream ss;
  ss << file.rdbuf();
  std::string err;
  const json11::Json json = json11::Json::parse(ss.str(), err);
  if (!err.empty() || !json.is_object()) return;
  for (const auto &[k, v] : json.object_items()) {
    if (v.is_string()) g_fingerprint_to_dbc[k] = v.string_value();
  }
}

void on_events_merged(const MessageEventsMap & /*unused*/) {
  if (can->liveStreaming()) return;
  const std::string fp = can->carFingerprint();
  if (fp == g_last_car_fingerprint) return;
  g_last_car_fingerprint = fp;

  // Don't overwrite an already-loaded DBC.
  if (dbc()->nonEmptyDBCCount() != 0) return;
  auto it = g_fingerprint_to_dbc.find(fp);
  if (it == g_fingerprint_to_dbc.end()) return;

  fprintf(stderr, "[cabana] auto-loading DBC for fingerprint '%s': %s.dbc\n", fp.c_str(), it->second.c_str());
  ui_load_dbc_from_opendbc(it->second + ".dbc");
}

// -- per-frame wiring ---------------------------------------------------------

void ensure_connected() {
  // Global objects (dbc(), UndoStack::instance()): connect once, keep the
  // once-guard.
  static bool connected = false;
  if (!connected) {
    connected = true;
    dbc()->DBCFileChanged.connect([]() {
      UndoStack::instance()->clear();
      mark_title_dirty();
    });
    UndoStack::instance()->cleanChanged.connect([](bool) { mark_title_dirty(); });

    load_fingerprints();
  }

  // The stream: File > Open Stream can swap `can` to a brand-new
  // AbstractStream at runtime (see stream_selector.cc's swap_stream()), so
  // rebind eventsMerged to whichever instance is current -- staying
  // connected to a torn-down stream is exactly why fingerprint auto-load
  // stops firing after a swap.
  static AbstractStream *wired_stream = nullptr;
  if (wired_stream != can) {
    wired_stream = can;
    can->eventsMerged.connect(on_events_merged);
    // The "already checked this fingerprint" latch is scoped to the
    // previous stream's car -- clear it so auto-load can fire again for
    // whatever route/car the new stream turns out to be.
    g_last_car_fingerprint.clear();
  }
}

// -- File menu: Manage DBC Files / Open Recent / Load from opendbc submenus
// (mirrors updateLoadSaveMenus / updateRecentFileMenu / createActions'
// opendbc loop) ---------------------------------------------------------------

void draw_manage_dbc_menu_contents() {
  for (int source : can->sources) {
    if (source >= 64) continue;  // Sent and blocked buses are handled implicitly
    const SourceSet ss = {source, uint8_t(source + 128), uint8_t(source + 192)};
    DBCFile *f = dbc()->findDBCFile(static_cast<uint8_t>(source));

    ImGui::PushID(source);
    const std::string bus_title = "Bus " + std::to_string(source) + " (" + (f != nullptr ? f->name() : "No DBCs loaded") + ")";
    if (ImGui::BeginMenu(bus_title.c_str())) {
      if (ImGui::MenuItem("New DBC File...")) ui_new_file(ss);
      if (ImGui::MenuItem("Open DBC File...")) ui_open_file(ss);
      if (ImGui::MenuItem("Load DBC From Clipboard...")) ui_load_from_clipboard(ss);
      if (f != nullptr) {
        ImGui::Separator();
        const std::string label = f->name() + " (" + toString(dbc()->sources(f)) + ")";
        ImGui::MenuItem(label.c_str(), nullptr, false, false);
        if (ImGui::MenuItem("Save...")) ui_save_file(f);
        if (ImGui::MenuItem("Save As...")) save_file_as(f, nullptr);
        if (ImGui::MenuItem("Copy to Clipboard...")) ui_copy_to_clipboard(f);
        if (ImGui::MenuItem("Remove from this bus...")) ui_close_file_sources(ss, nullptr);
        if (ImGui::MenuItem("Remove from all buses...")) ui_close_dbc_file(f);
      }
      ImGui::EndMenu();
    }
    ImGui::PopID();
  }
}

void draw_open_recent_menu() {
  if (!ImGui::BeginMenu("Open Recent")) return;
  const int n = std::min<int>(static_cast<int>(settings.recent_files.size()), MAX_RECENT_FILES);
  if (n == 0) {
    ImGui::MenuItem("No Recent Files", nullptr, false, false);
  } else {
    for (int i = 0; i < n; ++i) {
      const std::string &fn = settings.recent_files[static_cast<size_t>(i)];
      const std::string label = std::to_string(i + 1) + " " + fs::path(fn).filename().string();
      ImGui::PushID(i);
      if (ImGui::MenuItem(label.c_str())) ui_load_file(fn, SOURCE_ALL);
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", fn.c_str());
      ImGui::PopID();
    }
  }
  ImGui::EndMenu();
}

void draw_opendbc_menu() {
  if (!ImGui::BeginMenu("Load DBC from commaai/opendbc")) return;
  std::vector<std::string> names;
  std::error_code ec;
  fs::directory_iterator it(OPENDBC_FILE_PATH, ec);
  if (!ec) {
    for (const auto &entry : it) {
      if (!entry.is_regular_file()) continue;
      const std::string name = entry.path().filename().string();
      if (name.size() > 4 && name.compare(name.size() - 4, 4, ".dbc") == 0) names.push_back(name);
    }
  }
  std::sort(names.begin(), names.end());
  for (const auto &name : names) {
    if (ImGui::MenuItem(name.c_str())) ui_load_dbc_from_opendbc(name);
  }
  ImGui::EndMenu();
}

}  // namespace

// -- public API (dbc_menus.h) -------------------------------------------------

void dbc_menus_init(GLFWwindow *window) {
  g_window = window;
}

void dbc_menus_ensure_dbc_open() {
  if (dbc()->nonEmptyDBCCount() == 0) {
    dbc()->open(SOURCE_ALL, std::string(), std::string());
  }
}

void dbc_menus_update() {
  ensure_connected();
  refresh_window_title_if_dirty();

  const ImGuiIO &io = ImGui::GetIO();
  if (io.WantTextInput || !(io.KeyCtrl || io.KeySuper)) return;

  UndoStack *us = UndoStack::instance();
  if (ImGui::IsKeyPressed(ImGuiKey_N, false)) {
    ui_new_file(SOURCE_ALL);
  } else if (ImGui::IsKeyPressed(ImGuiKey_O, false)) {
    ui_open_file(SOURCE_ALL);
  } else if (io.KeyShift && ImGui::IsKeyPressed(ImGuiKey_S, false)) {
    ui_save_as_all();
  } else if (ImGui::IsKeyPressed(ImGuiKey_S, false)) {
    ui_save_all();
  } else if (io.KeyShift && ImGui::IsKeyPressed(ImGuiKey_Z, false)) {
    if (us->canRedo()) us->redo();
  } else if (ImGui::IsKeyPressed(ImGuiKey_Z, false)) {
    if (us->canUndo()) us->undo();
  }
}

void draw_dbc_file_menu_items(AppState &app) {
  ensure_connected();

  if (ImGui::MenuItem("New DBC File", "Ctrl+N")) ui_new_file(SOURCE_ALL);
  if (ImGui::MenuItem("Open DBC File...", "Ctrl+O")) ui_open_file(SOURCE_ALL);

  if (ImGui::BeginMenu("Manage DBC Files", has_stream(app))) {
    draw_manage_dbc_menu_contents();
    ImGui::EndMenu();
  }
  draw_open_recent_menu();

  ImGui::Separator();
  draw_opendbc_menu();
  if (ImGui::MenuItem("Load DBC From Clipboard")) ui_load_from_clipboard(SOURCE_ALL);

  ImGui::Separator();
  const int cnt = dbc()->nonEmptyDBCCount();
  const std::string save_label = cnt > 1 ? ("Save " + std::to_string(cnt) + " DBCs...") : std::string("Save DBC...");
  if (ImGui::MenuItem(save_label.c_str(), "Ctrl+S", false, cnt > 0)) ui_save_all();
  if (ImGui::MenuItem("Save DBC As...", "Ctrl+Shift+S", false, cnt == 1)) ui_save_as_all();
  if (ImGui::MenuItem("Copy DBC To Clipboard", nullptr, false, cnt == 1)) ui_copy_to_clipboard_all();
}

void draw_dbc_edit_menu_items() {
  UndoStack *us = UndoStack::instance();

  std::string undo_label = "Undo";
  if (us->canUndo()) undo_label += " " + us->undoText();
  if (ImGui::MenuItem(undo_label.c_str(), "Ctrl+Z", false, us->canUndo())) us->undo();

  std::string redo_label = "Redo";
  if (us->canRedo()) redo_label += " " + us->redoText();
  if (ImGui::MenuItem(redo_label.c_str(), "Ctrl+Shift+Z", false, us->canRedo())) us->redo();

  ImGui::Separator();
  if (ImGui::BeginMenu("Command List")) {
    // mirrors QUndoView: row 0 is the pre-history "<empty>" state (index 0),
    // then one row per command (index i+1 after it's applied).
    if (ImGui::MenuItem("<empty>", nullptr, us->index() == 0)) us->setIndex(0);
    for (int i = 0; i < us->count(); ++i) {
      ImGui::PushID(i);
      if (ImGui::MenuItem(us->text(i).c_str(), nullptr, us->index() == i + 1)) us->setIndex(i + 1);
      ImGui::PopID();
    }
    ImGui::EndMenu();
  }
}

void draw_dbc_modals() {
  file_dialog_draw();
  draw_remind_modal();
  draw_simple_modal();
}

void dbc_menus_begin_close() {
  remind_save_changes([]() {
    if (g_window != nullptr) glfwSetWindowShouldClose(g_window, GLFW_TRUE);
  });
}

void dbc_menus_note_recent_file(const std::string &fn) {
  note_recent_file(fn);
}
