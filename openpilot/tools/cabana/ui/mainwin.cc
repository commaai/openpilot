#include "tools/cabana/ui/mainwin.h"

#include <algorithm>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "imgui.h"
#include "imgui_internal.h"
#include <GLFW/glfw3.h>

#include "json11/json11.hpp"
#include "tools/cabana/commands.h"
#include "tools/cabana/settings.h"
#include "tools/cabana/ui/app.h"
#include "tools/cabana/ui/dialogs/filedialog.h"
#include "tools/cabana/ui/dialogs/messagebox.h"
#include "tools/cabana/ui/inistate.h"
#include "tools/cabana/ui/threadpool.h"
#include "tools/cabana/ui/tools/findsignal.h"
#include "tools/cabana/ui/tools/findsimilarbits.h"
#include "tools/cabana/ui/util.h"
#include "tools/cabana/utils/export.h"
#include "tools/cabana/utils/util.h"
#include "tools/replay/py_downloader.h"
#include "tools/replay/util.h"

namespace {
// dock window ids (the visible titles change, the part after ### is the identity)
constexpr const char *VIDEO_PANEL = "###VideoPanel";
constexpr const char *CENTER_PANEL = "CAN Details###CenterWidget";
constexpr const char *CHARTS_PANEL = "Charts###ChartsPanel";
constexpr const char *LOG_MESSAGES_PANEL = "openpilot Messages###LogMessagesPanel";
}  // namespace

MainWindow::MainWindow(GLFWwindow *window, std::unique_ptr<AbstractStream> stream, StreamLoader stream_loader,
                       const std::string &dbc_file, const std::string &layout) : startup_layout_(layout), window_(window) {
  can = &dummy_;
  details_visible_ = inistate::main_window.details_visible;
  // Replace the old combined Messages pane with independently dockable source panels.
  reset_layout_ = inistate::main_window.workspace_version < 2;
  messages_visible_ = inistate::main_window.messages_visible;
  log_messages_visible_ = inistate::main_window.log_messages_visible;
  charts_visible_ = inistate::main_window.charts_visible;
  video_visible_ = inistate::main_window.video_visible;
  loadFingerprints();
  std::error_code ec;
  for (const auto &entry : std::filesystem::directory_iterator(OPENDBC_FILE_PATH, ec)) {
    if (entry.is_regular_file() && entry.path().extension() == ".dbc") {
      opendbc_names_.push_back(entry.path().filename().string());
    }
  }
  std::sort(opendbc_names_.begin(), opendbc_names_.end());

  // download handlers are called from download threads
  installDownloadProgressHandler([this](uint64_t cur, uint64_t total, bool success) {
    utils::runOnMainThread([this, cur, total, success]() { updateDownloadProgress(cur, total, success); });
  });
  installMessageHandler([this](ReplyMsgType type, const std::string &msg) {
    utils::runOnMainThread([this, msg]() { showStatusMessage(msg, 2000); });
  });

  connections_.push_back(dbc()->fileChanged.connect([this]() { dbcFileChanged(); }));
  connections_.push_back(UndoStack::instance()->cleanChanged.connect([this](bool clean) {
    window_modified_ = !clean;
    updateWindowTitle();
  }));

  startup_stream_ = std::move(stream);
  startup_loader_ = std::move(stream_loader);
  nextFrame([this, dbc_file]() {
    if (startup_loader_) {
      loadStartupStream(dbc_file);
    } else {
      startup_stream_ ? openStream(std::move(startup_stream_), dbc_file) : selectAndOpenStream();
    }
  });
}

void MainWindow::loadFingerprints() {
  std::ifstream json_file((executableDir() / "dbc/car_fingerprint_to_dbc.json"));
  if (!json_file) return;
  const std::string contents{std::istreambuf_iterator<char>(json_file), std::istreambuf_iterator<char>()};
  std::string err;
  auto doc = json11::Json::parse(contents, err);
  if (!err.empty() || !doc.is_object()) return;
  for (const auto &kv : doc.object_items()) {
    if (kv.second.is_string()) {
      fingerprint_to_dbc_.emplace(kv.first, kv.second.string_value());
    }
  }
}

void MainWindow::drawFileMenu() {
  const bool has_stream = hasStream();
  if (ImGui::MenuItem("Open Stream...")) selectAndOpenStream();
  if (ImGui::MenuItem("Close Stream", nullptr, false, has_stream)) closeStream();
  if (ImGui::MenuItem("Export to CSV...", nullptr, false, has_stream)) exportToCSV();
  ImGui::Separator();

  if (ImGui::MenuItem("New DBC File", shortcut("N").c_str())) newFile();
  if (ImGui::MenuItem("Open DBC File...", shortcut("O").c_str())) openFile();

  if (ImGui::BeginMenu("Manage DBC Files", has_stream)) {
    drawManageDBCsMenu();
    ImGui::EndMenu();
  }
  if (ImGui::BeginMenu("Open Recent")) {
    drawRecentFilesMenu();
    ImGui::EndMenu();
  }

  ImGui::Separator();
  if (ImGui::BeginMenu("Load DBC from commaai/opendbc")) {
    for (const auto &name : opendbc_names_) {
      if (ImGui::MenuItem(name.c_str())) loadDBCFromOpendbc(name);
    }
    ImGui::EndMenu();
  }
  if (ImGui::MenuItem("Load DBC from Clipboard")) loadFromClipboard();

  ImGui::Separator();
  const int cnt = dbc()->nonEmptyDBCCount();
  const std::string save_text = cnt > 1 ? "Save " + std::to_string(cnt) + " DBCs..." : "Save DBC...";
  if (ImGui::MenuItem(save_text.c_str(), shortcut("S").c_str(), false, cnt > 0)) save();
  if (ImGui::MenuItem("Save DBC As...", shortcut("Shift+S").c_str(), false, cnt == 1)) saveAs();
  // TODO: Support clipboard for multiple files
  if (ImGui::MenuItem("Copy DBC to Clipboard", nullptr, false, cnt == 1)) saveToClipboard();

  ImGui::Separator();
  if (ImGui::MenuItem("Settings...")) openSettings();

  ImGui::Separator();
  if (ImGui::MenuItem("Exit", shortcut("Q").c_str())) close();
}

namespace {
bool beginTopMenu(const char *label, bool enabled = true) {
  ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImGui::GetColorU32(ImGuiCol_Header));
  const bool open = ImGui::BeginMenu(label, enabled);
  ImGui::PopStyleColor();
  if (open) {
    // Erase the popup's rounded top border so it joins the menu bar's separator.
    const ImGuiStyle &style = ImGui::GetStyle();
    const ImGuiWindow *w = ImGui::GetCurrentWindow();
    const float r = style.PopupRounding, b = style.PopupBorderSize;
    const ImVec2 min = w->Pos, max(w->Pos.x + w->Size.x, w->Pos.y + w->Size.y);
    const ImU32 bg = ImGui::GetColorU32(ImGuiCol_PopupBg), border = ImGui::GetColorU32(ImGuiCol_Border);
    ImDrawList *dl = w->DrawList;
    dl->PushClipRect(min, max, false);  // the window's own clip rect excludes its border
    dl->AddRectFilled(min, ImVec2(max.x, min.y + r), bg);
    dl->AddRectFilled(ImVec2(min.x, min.y), ImVec2(min.x + b, min.y + r), border);
    dl->AddRectFilled(ImVec2(max.x - b, min.y), ImVec2(max.x, min.y + r), border);
    dl->PopClipRect();
  }
  return open;
}
}  // namespace

void MainWindow::drawMenuBar() {
  // Avoid a double border with the separator drawn below.
  ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
  const bool open = ImGui::BeginMainMenuBar();
  ImGui::PopStyleVar();
  if (!open) return;
  {
    const ImVec2 min = ImGui::GetWindowPos();
    const ImVec2 max(min.x + ImGui::GetWindowWidth(), min.y + ImGui::GetWindowHeight());
    ImGui::GetWindowDrawList()->AddRectFilled(ImVec2(min.x, max.y - 1.0f), max, ImGui::GetColorU32(ImGuiCol_Border));
  }
  if (beginTopMenu("File")) {
    drawFileMenu();
    ImGui::EndMenu();
  }

  if (beginTopMenu("Edit")) {
    auto stack = UndoStack::instance();
    const std::string undo_text = stack->canUndo() ? "Undo " + stack->undoText() : "Undo";
    const std::string redo_text = stack->canRedo() ? "Redo " + stack->redoText() : "Redo";
    if (ImGui::MenuItem(undo_text.c_str(), shortcut("Z").c_str(), false, stack->canUndo())) stack->undo();
    if (ImGui::MenuItem(redo_text.c_str(), shortcut("Shift+Z").c_str(), false, stack->canRedo())) stack->redo();
    ImGui::EndMenu();
  }

  if (beginTopMenu("View")) {
    if (ImGui::MenuItem("Full Screen", shortcut("F11").c_str())) toggleFullScreen();
    ImGui::Separator();
    ImGui::MenuItem("CAN signals", nullptr, &messages_visible_);
    ImGui::MenuItem("openpilot Messages", nullptr, &log_messages_visible_);
    ImGui::MenuItem("Charts", nullptr, &charts_visible_);
    ImGui::MenuItem("CAN Details", nullptr, &details_visible_);
    ImGui::MenuItem("Playback", nullptr, &video_visible_);
    ImGui::Separator();
    if (ImGui::MenuItem("Reset Window Layout")) {
      messages_visible_ = log_messages_visible_ = charts_visible_ = video_visible_ = true;
      reset_layout_ = true;
    }
    ImGui::EndMenu();
  }

  if (beginTopMenu("Tools", hasStream())) {
    if (ImGui::MenuItem("Find Similar Bits")) findSimilarBits();
    if (ImGui::MenuItem("Find Signal")) findSignal();
    ImGui::EndMenu();
  }

  if (beginTopMenu("Help")) {
    if (ImGui::MenuItem("Help", "F1")) toggleHelp();
    ImGui::EndMenu();
  }
  ImGui::EndMainMenuBar();
}

void MainWindow::createDockWidgets() {
  widget_connections_.clear();
  messages_widget_ = std::make_unique<MessagesWidget>();
  widget_connections_.push_back(messages_widget_->msgSelectionChanged.connect([this](const MessageId &id) { showMessage(id); }));

  charts_widget_ = std::make_unique<ChartsWidget>();
  center_widget_.setChartsWidget(charts_widget_.get());
  widget_connections_.push_back(charts_widget_->showLogMessages.connect([this]() {
    log_messages_visible_ = true;
    selectPanelTab(LOG_MESSAGES_PANEL);
  }));
  video_widget_ = std::make_unique<VideoWidget>();
  widget_connections_.push_back(charts_widget_->chartAdded.connect([this]() {
    charts_visible_ = true;
    selectPanelTab(CHARTS_PANEL);
  }));
}

void MainWindow::showStatusMessage(const std::string &msg, int timeout_ms) {
  status_bar_.message = msg;
  status_bar_.message_until = timeout_ms > 0 ? ImGui::GetTime() + timeout_ms / 1000.0 : 0;
}

void MainWindow::updateWindowTitle() {
  std::string title;
  for (auto f : dbc()->allDBCFiles()) {
    if (!title.empty()) title += " | ";
    title += "(" + toString(dbc()->sources(f)) + ") " + f->name();
  }
  if (window_modified_) title += "*";
  if (!title.empty()) title += " \xe2\x80\x94 ";  // em dash separator
  title += "Cabana";
  glfwSetWindowTitle(window_, title.c_str());
}

void MainWindow::dbcFileChanged() {
  UndoStack::instance()->clear();
  updateWindowTitle();
  nextFrame([this]() { restoreSessionState(); });
}

void MainWindow::selectAndOpenStream() {
  stream_selector_.open([this](std::unique_ptr<AbstractStream> stream, const std::string &dbc_file) {
    if (stream) {
      openStream(std::move(stream), dbc_file);
    } else if (!stream_) {
      openStream(std::make_unique<DummyStream>());
    }
  });
}

// the route file listing hits the comma API, so the loader runs on a worker behind the window
void MainWindow::loadStartupStream(const std::string &dbc_file) {
  wait_dlg_.text = "Loading route...";
  wait_dlg_.value = 0;
  wait_dlg_.open = true;
  wait_dlg_.show_at = ImGui::GetTime() + 4.0;  // minimum duration before the dialog shows
  ThreadPool::instance().run([this, dbc_file, loader = std::move(startup_loader_)]() {
    AbstractStream *loaded = nullptr;
    std::string error;
    try {
      loaded = loader().release();
    } catch (const std::exception &e) {
      // the pool swallows exceptions, so the wait dialog would spin forever
      error = e.what();
    }
    utils::runOnMainThread([this, dbc_file, loaded, error]() {
      wait_dlg_.open = false;
      std::unique_ptr<AbstractStream> stream(loaded);
      if (!error.empty()) {
        fprintf(stderr, "%s\n", error.c_str());
        MessageBox::warning("Failed to load route", error);
      }
      stream ? openStream(std::move(stream), dbc_file) : openStream(std::make_unique<DummyStream>());
    });
  });
}

void MainWindow::closeStream() {
  openStream(std::make_unique<DummyStream>());
  if (dbc()->nonEmptyDBCCount() > 0) {
    dbc()->fileChanged();
  }
  showStatusMessage("Stream closed");
}

void MainWindow::exportToCSV() {
  std::string dir = settings.last_dir + "/" + can->routeName() + ".csv";
  FileDialog::getSaveFileName("Export stream to CSV file", dir, ".csv", [](const std::string &fn) {
    if (!fn.empty()) {
      utils::exportToCSV(fn);
    }
  });
}

void MainWindow::newFile(SourceSet s) {
  closeFile(s, [s]() { dbc()->open(s, std::string(""), std::string("")); });
}

void MainWindow::openFile(SourceSet s) {
  remindSaveChanges([this, s]() {
    FileDialog::getOpenFileName("Open File", settings.last_dir, ".dbc", [this, s](const std::string &fn) {
      if (!fn.empty()) {
        loadFile(fn, s);
      }
    });
  });
}

void MainWindow::loadFile(const std::string &fn, SourceSet s, std::function<void()> then) {
  if (!fn.empty()) {
    closeFile(s, [this, fn, s, then]() {
      std::string error;
      if (dbc()->open(s, fn, &error)) {
        updateRecentFiles(fn);
        showStatusMessage("DBC file " + fn + " loaded", 2000);
        if (then) then();
      } else {
        MessageBox::warning("Failed to load DBC file", "Failed to parse DBC file " + fn, error, then);
      }
    });
  } else if (then) {
    then();
  }
}

void MainWindow::loadDBCFromOpendbc(const std::string &name) {
  loadFile(std::string(OPENDBC_FILE_PATH) + "/" + name);
}

void MainWindow::loadFromClipboard(SourceSet s, bool close_all) {
  std::string text;
  if (!utils::getClipboardText(&text)) {
    MessageBox::warning("Load from Clipboard", "No clipboard tool found. Install xclip (X11) or wl-clipboard (Wayland).");
    return;
  }
  if (text.empty()) {
    MessageBox::warning("Load from Clipboard", "Clipboard is empty.");
    return;
  }

  closeFile(s, [s, text]() {
    std::string error;
    bool ret = dbc()->open(s, std::string(""), text, &error);
    if (ret && dbc()->nonEmptyDBCCount() > 0) {
      MessageBox::information("Load from Clipboard", "DBC loaded successfully.");
    } else {
      MessageBox::warning("Failed to load DBC from clipboard", "Make sure the clipboard contains correctly formatted DBC text.", error);
    }
  });
}

MainWindow::~MainWindow() {
  installDownloadProgressHandler(nullptr);
  installMessageHandler(nullptr);
  releaseStream();
  can = nullptr;
}

// the tool dialogs are connected into the messages widget, and the video widget's RouteInfoDlg keeps a raw
// pointer to the replay, so the dialogs go first and the widgets before the stream; the stream's destructor
// joins the threads that read the global `can`
void MainWindow::releaseStream() {
  tool_dialogs_.clear();
  wait_dlg_.connection.disconnect();
  wait_dlg_.open = false;
  widget_connections_.clear();
  charts_widget_.reset();
  video_widget_.reset();
  center_widget_.clear();
  messages_widget_.reset();
  stream_connections_.clear();
  stream_.reset();
  can = &dummy_;
}

void MainWindow::openStream(std::unique_ptr<AbstractStream> stream, const std::string &dbc_file) {
  releaseStream();
  startStream(std::move(stream), dbc_file);
}

void MainWindow::startStream(std::unique_ptr<AbstractStream> stream, const std::string &dbc_file) {
  stream_ = std::move(stream);
  can = stream_.get();
  stream_connections_.push_back(can->error.connect([](const std::string &msg) {
    MessageBox::warning("Error", msg);
  }));
  can->start();

  loadFile(dbc_file, SOURCE_ALL, [this]() {
    showStatusMessage("Stream [" + can->routeName() + "] started", 2000);
    createDockWidgets();
    nextFrame([this]() { restoreSessionState(); });

    video_dock_title_ = can->routeName();
    // Don't overwrite already loaded DBC
    if (!dbc()->nonEmptyDBCCount()) {
      newFile();
    }

    stream_connections_.push_back(can->fieldsChanged.connect([this]() { wait_dlg_.open = false; }));
    stream_connections_.push_back(can->eventsMerged.connect([this](const MessageEventsMap &) { eventsMerged(); }));

    if (hasStream()) {
      wait_dlg_.text = can->liveStreaming() ? "Waiting for the live stream to start..." : "Loading segment data...";
      wait_dlg_.value = 0;
      wait_dlg_.open = true;
      wait_dlg_.show_at = ImGui::GetTime() + 4.0;  // minimum duration before the dialog shows
      wait_dlg_.connection = can->eventsMerged.connect([this](const MessageEventsMap &) {
        wait_dlg_.open = false;
        wait_dlg_.connection.disconnect();
      });
    }
  });
}

void MainWindow::eventsMerged() {
  const std::string fingerprint = can->carFingerprint();
  if (!can->liveStreaming() && std::exchange(car_fingerprint_, fingerprint) != fingerprint) {
    video_dock_title_ = "ROUTE: " + can->routeName() + "  FINGERPRINT: " + (car_fingerprint_.empty() ? "Unknown Car" : car_fingerprint_);
    // Don't overwrite already loaded DBC
    auto it = fingerprint_to_dbc_.find(car_fingerprint_);
    if (!dbc()->nonEmptyDBCCount() && it != fingerprint_to_dbc_.end()) {
      nextFrame([this, dbc_name = it->second]() { loadDBCFromOpendbc(dbc_name + ".dbc"); });
    }
  }
}

void MainWindow::saveFiles(bool as, std::function<void()> then) {
  const std::vector<DBCFile *> files = dbc()->nonEmptyDBCFiles();
  auto next = std::make_shared<std::function<void(size_t)>>();
  *next = [this, as, files, next, then](size_t i) {
    if (i >= files.size()) {
      if (then) then();
      return;
    }
    auto cb = [next, i]() { (*next)(i + 1); };
    as ? saveFileAs(files[i], cb) : saveFile(files[i], cb);
  };
  (*next)(0);
}

void MainWindow::save(std::function<void()> then) {
  saveFiles(false, std::move(then));
}

void MainWindow::saveAs(std::function<void()> then) {
  saveFiles(true, std::move(then));
}

void MainWindow::closeFile(SourceSet s, std::function<void()> then) {
  remindSaveChanges([s, then]() {
    if (s == SOURCE_ALL) {
      dbc()->closeAll();
    } else {
      dbc()->close(s);
    }
    if (then) then();
  });
}

void MainWindow::closeFile(DBCFile *dbc_file) {
  assert(dbc_file != nullptr);
  remindSaveChanges([this, dbc_file]() {
    dbc()->close(dbc_file);
    // Ensure we always have at least one file open
    if (dbc()->dbcCount() == 0) {
      newFile();
    }
  });
}

void MainWindow::saveFile(DBCFile *dbc_file, std::function<void()> then) {
  assert(dbc_file != nullptr);
  if (!dbc_file->filename.empty()) {
    dbc_file->save();
    UndoStack::instance()->setClean();
    showStatusMessage("File saved", 2000);
    if (then) then();
  } else if (!dbc_file->isEmpty()) {
    saveFileAs(dbc_file, then);
  } else if (then) {
    then();
  }
}

void MainWindow::saveFileAs(DBCFile *dbc_file, std::function<void()> then) {
  std::string title = "Save File (bus: " + toString(dbc()->sources(dbc_file)) + ")";
  std::string default_path = (std::filesystem::path(settings.last_dir) / "untitled.dbc").string();
  FileDialog::getSaveFileName(title, default_path, ".dbc", [this, dbc_file, then](const std::string &fn) {
    if (!fn.empty()) {
      dbc_file->saveAs(fn);
      UndoStack::instance()->setClean();
      showStatusMessage("File saved as " + fn, 2000);
      updateRecentFiles(fn);
    }
    if (then) then();
  });
}

void MainWindow::saveToClipboard() {
  // Should not be called with more than 1 file open
  for (auto dbc_file : dbc()->nonEmptyDBCFiles()) {
    saveFileToClipboard(dbc_file);
  }
}

void MainWindow::saveFileToClipboard(DBCFile *dbc_file) {
  assert(dbc_file != nullptr);
  copyToClipboard(dbc_file->generateDBC());
}

void MainWindow::copyToClipboard(const std::string &text) {
  if (utils::setClipboardText(text)) {
    MessageBox::information("Copy to Clipboard", "DBC copied successfully.");
  } else {
    MessageBox::warning("Copy to Clipboard", "Failed to copy DBC to clipboard. Install xclip (X11) or wl-clipboard (Wayland).");
  }
}

void MainWindow::drawManageDBCsMenu() {
  for (int source : can->sources) {
    if (source >= 64) continue; // Sent and blocked buses are handled implicitly

    SourceSet ss = {source, uint8_t(source + 128), uint8_t(source + 192)};

    auto dbc_file = dbc()->findDBCFile(source);
    const std::string title = "Bus " + std::to_string(source) + " (" + (dbc_file ? dbc_file->name() : "No DBCs loaded") + ")";
    ImGui::PushID(source);
    if (ImGui::BeginMenu(title.c_str())) {
      if (ImGui::MenuItem("New DBC File")) newFile(ss);
      if (ImGui::MenuItem("Open DBC File...")) openFile(ss);
      if (ImGui::MenuItem("Load DBC from Clipboard")) loadFromClipboard(ss, false);

      // Show sub-menu for each dbc for this source.
      if (dbc_file) {
        ImGui::Separator();
        ImGui::MenuItem((dbc_file->name() + " (" + toString(dbc()->sources(dbc_file)) + ")").c_str(), nullptr, false, false);
        if (ImGui::MenuItem("Save...")) saveFile(dbc_file);
        if (ImGui::MenuItem("Save As...")) saveFileAs(dbc_file);
        if (ImGui::MenuItem("Copy to Clipboard")) saveFileToClipboard(dbc_file);
        if (ImGui::MenuItem("Remove from This Bus...")) closeFile(ss, {});
        if (ImGui::MenuItem("Remove from All Buses...")) closeFile(dbc_file);
      }
      ImGui::EndMenu();
    }
    ImGui::PopID();
  }
}

void MainWindow::updateRecentFiles(const std::string &fn) {
  settings.recent_files.erase(std::remove(settings.recent_files.begin(), settings.recent_files.end(), fn), settings.recent_files.end());
  settings.recent_files.insert(settings.recent_files.begin(), fn);
  while (settings.recent_files.size() > MAX_RECENT_FILES) {
    settings.recent_files.pop_back();
  }
  settings.last_dir = std::filesystem::absolute(fn).parent_path().string();
}

void MainWindow::drawRecentFilesMenu() {
  int num_recent_files = std::min<int>(settings.recent_files.size(), MAX_RECENT_FILES);
  if (!num_recent_files) {
    ImGui::MenuItem("No Recent Files", nullptr, false, false);
    return;
  }

  for (int i = 0; i < num_recent_files; ++i) {
    std::string text = std::to_string(i + 1) + " " + std::filesystem::path(settings.recent_files[i]).filename().string();
    ImGui::PushID(i);
    if (ImGui::MenuItem(text.c_str())) loadFile(settings.recent_files[i]);
    ImGui::PopID();
  }
}

void MainWindow::remindSaveChanges(std::function<void()> then) {
  if (UndoStack::instance()->isClean()) {
    UndoStack::instance()->clear();
    if (then) then();
    return;
  }
  std::string text = "You have unsaved changes. Select OK to save them or Cancel to discard them.";
  MessageBox::question("Unsaved Changes", text, [this, then](bool ok) {
    if (ok) {
      save([this, then]() { remindSaveChanges(then); });
    } else {
      UndoStack::instance()->clear();
      if (then) then();
    }
  });
}

void MainWindow::updateDownloadProgress(uint64_t cur, uint64_t total, bool success) {
  const double fraction = total > 0 ? cur / (double)total : 0.0;
  if (wait_dlg_.open) wait_dlg_.value = (int)(fraction * 100);
  if (success && cur < total) {
    status_bar_.progress_value = fraction;
    status_bar_.progress_text = "Downloading " + std::to_string((int)(fraction * 100)) + "% (" + formattedDataSize(total) + ")";
    status_bar_.progress_visible = true;
  } else {
    status_bar_.progress_visible = false;
  }
}

void MainWindow::close() {
  if (closing_) return;
  closing_ = true;
  remindSaveChanges([this]() { finishClose(); });
}

void MainWindow::finishClose() {
  // save states
  auto &state = inistate::main_window;
  state.maximized = glfwGetWindowAttrib(window_, GLFW_MAXIMIZED);
  if (full_screen_) {
#ifndef __APPLE__
    // macOS full screen is the native Cocoa toggle, keep the loaded geometry there
    state.pos[0] = windowed_rect_[0]; state.pos[1] = windowed_rect_[1];
    state.size[0] = windowed_rect_[2]; state.size[1] = windowed_rect_[3];
#endif
  } else if (!state.maximized) {
    glfwGetWindowPos(window_, &state.pos[0], &state.pos[1]);
    glfwGetWindowSize(window_, &state.size[0], &state.size[1]);
  }
  state.has_geometry = state.size[0] > 0 && state.size[1] > 0;
  state.workspace_version = 2;
  state.log_messages_visible = log_messages_visible_;
  state.charts_visible = charts_visible_;
  state.details_visible = details_visible_;
  state.messages_visible = messages_visible_;
  state.video_visible = video_visible_;
  settings.ui_state = inistate::save();

  saveSessionState();
  settings.save();
  exited_ = true;
}

void MainWindow::openSettings() {
  settings_dialog_.open();
}

void MainWindow::findSimilarBits() {
  auto dlg = std::make_unique<FindSimilarBitsDlg>();
  dlg->connections_.push_back(dlg->openMessage.connect([this](const MessageId &id) { messages_widget_->selectMessage(id); }));
  tool_dialogs_.push_back(std::move(dlg));
}

void MainWindow::findSignal() {
  auto dlg = std::make_unique<FindSignalDlg>();
  dlg->connections_.push_back(dlg->openMessage.connect([this](const MessageId &id) { messages_widget_->selectMessage(id); }));
  tool_dialogs_.push_back(std::move(dlg));
}

void MainWindow::toggleHelp() {
  help_overlay_.toggle();
}

void MainWindow::toggleFullScreen() {
#ifdef __APPLE__
  toggleNativeFullScreen(window_);
#else
  full_screen_ = !full_screen_;
  if (full_screen_) {
    glfwGetWindowPos(window_, &windowed_rect_[0], &windowed_rect_[1]);
    glfwGetWindowSize(window_, &windowed_rect_[2], &windowed_rect_[3]);
    GLFWmonitor *monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode *mode = glfwGetVideoMode(monitor);
    glfwSetWindowMonitor(window_, monitor, 0, 0, mode->width, mode->height, mode->refreshRate);
  } else {
    glfwSetWindowMonitor(window_, nullptr, windowed_rect_[0], windowed_rect_[1], windowed_rect_[2], windowed_rect_[3], 0);
    glfwMaximizeWindow(window_);
  }
#endif
}

void MainWindow::saveSessionState() {
  settings.recent_dbc_file = "";
  settings.active_msg_id = "";
  settings.selected_msg_ids.clear();

  const auto files = dbc()->nonEmptyDBCFiles();
  if (!files.empty()) settings.recent_dbc_file = files.front()->filename;

  if (auto *detail = center_widget_.getDetailWidget()) {
    auto [active_id, ids] = detail->serializeMessageIds();
    settings.active_msg_id = active_id;
    settings.selected_msg_ids = ids;
  }
}

void MainWindow::restoreSessionState() {
  if (!charts_widget_) return;
  // CAN layouts may need the DBC loaded by eventsMerged(). dbcFileChanged() retries while definitions are missing.
  if (!startup_layout_.empty()) {
    if (charts_widget_->openLayout(startup_layout_, true) != ChartsWidget::LayoutStatus::MissingCan) startup_layout_.clear();
  }
  if (settings.recent_dbc_file.empty() || dbc()->nonEmptyDBCCount() == 0) return;

  if (dbc()->nonEmptyDBCFiles().front()->filename != settings.recent_dbc_file) return;

  if (!settings.selected_msg_ids.empty()) {
    center_widget_.ensureDetailWidget()->restoreTabs(settings.active_msg_id, settings.selected_msg_ids);
  }
}

void MainWindow::handleShortcuts() {
  const ImGuiIO &io = ImGui::GetIO();
  for (const KeyEvent &e : takeKeyEvents()) {
    const bool ctrl = e.mods & (GLFW_MOD_CONTROL | GLFW_MOD_SUPER);
    const bool shift = e.mods & GLFW_MOD_SHIFT;
    // a focused text input consumes Space but not the Ctrl/F-key sequences
    if (e.key == GLFW_KEY_SPACE && !ctrl && can && !io.WantTextInput) can->pause(!can->isPaused());
    if (e.key == GLFW_KEY_F1) toggleHelp();
    if (e.key == GLFW_KEY_F11 && ctrl) toggleFullScreen();
    // an open popup or a focused text input takes Esc first
    if (e.key == GLFW_KEY_ESCAPE && full_screen_ && !io.WantTextInput &&
        !ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel)) {
      toggleFullScreen();
    }
    if (!ctrl) continue;
    if (e.key == GLFW_KEY_N) newFile();
    if (e.key == GLFW_KEY_O) openFile();
    if (e.key == GLFW_KEY_S) {
      if (shift) {
        if (dbc()->nonEmptyDBCCount() == 1) saveAs();
      } else if (dbc()->nonEmptyDBCCount() > 0) {
        save();
      }
    }
    // a focused text input swallows Ctrl+Z / Ctrl+Shift+Z
    if (e.key == GLFW_KEY_Z && !io.WantTextInput) shift ? UndoStack::instance()->redo() : UndoStack::instance()->undo();
    if (e.key == GLFW_KEY_Q) close();
  }
}

void MainWindow::drawStatusBar() {
  ImGui::PushStyleColor(ImGuiCol_ChildBg, ImGui::GetStyle().Colors[ImGuiCol_MenuBarBg]);
  ImGui::BeginChild("status_bar", ImVec2(0, ImGui::GetFrameHeight()), ImGuiChildFlags_None, ImGuiWindowFlags_NoScrollbar);
  const ImVec2 min = ImGui::GetWindowPos();
  ImGui::GetWindowDrawList()->AddRectFilled(min, ImVec2(min.x + ImGui::GetWindowWidth(), min.y + 1.0f), ImGui::GetColorU32(ImGuiCol_Border));
  // a borderless child gets no WindowPadding, so both ends sit flush against the edge and clip. Inset by
  // WindowPadding.x, which lines the text up with the content of the docked panels above (the messages table).
  const float width = ImGui::GetContentRegionAvail().x;
  const float pad = ImGui::GetStyle().WindowPadding.x;
  ImGui::SetCursorPosX(pad);
  ImGui::AlignTextToFramePadding();
  // a temporary message hides the normal widgets, permanent widgets stay on the right
  auto &bar = status_bar_;
  if (!bar.message.empty() && (bar.message_until == 0 || ImGui::GetTime() < bar.message_until)) {
    ImGui::TextUnformatted(bar.message.c_str());
  } else {
    bar.message.clear();
    ImGui::TextUnformatted("For help, press F1");
  }
  if (bar.progress_visible) {
    ImGui::SameLine(width - pad - 300.0f);
    ImGui::ProgressBar(bar.progress_value, ImVec2(300.0f, 16.0f), bar.progress_text.c_str());
  }
  ImGui::EndChild();
  ImGui::PopStyleColor();
}

void MainWindow::drawWaitDialog() {
  const char *id = "###WaitDialog";
  if (wait_dlg_.open && !ImGui::IsPopupOpen(id) && ImGui::GetTime() >= wait_dlg_.show_at) ImGui::OpenPopup(id);
  if (!ImGui::IsPopupOpen(id)) return;  // keep submitting until CloseCurrentPopup ran, a stale modal blocks all input
  ImGui::SetNextWindowSize(ImVec2(400.0f, 0.0f), ImGuiCond_Always);
  setNextDialogWindow(ImVec2(0.0f, 0.0f));
  if (ImGui::BeginPopupModal(id, nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::TextUnformatted(wait_dlg_.text.c_str());
    // no text until the progress is set
    ImGui::ProgressBar(wait_dlg_.value / 100.0f, ImVec2(-1.0f, 0.0f), wait_dlg_.value == 0 ? "" : (const char *)nullptr);
    bool abort = false, rejected = false;
    dialogButtons("Abort", &abort, &rejected, true, nullptr);
    if (abort || rejected) {
      wait_dlg_.open = false;
      close();
    }
    if (!wait_dlg_.open) ImGui::CloseCurrentPopup();
    ImGui::EndPopup();
  }
}

void MainWindow::drawDockspace() {
  const ImGuiViewport *viewport = ImGui::GetMainViewport();
  // Use the menu bar's current-frame reservation, including on the first frame.
  const ImRect work_rect = static_cast<const ImGuiViewportP *>(viewport)->GetBuildWorkRect();
  ImGui::SetNextWindowPos(work_rect.Min);
  ImGui::SetNextWindowSize(work_rect.GetSize());
  ImGui::SetNextWindowViewport(viewport->ID);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
  const ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize |
                                 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus |
                                 ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoBackground |
                                 ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse;
  ImGui::Begin("##host", nullptr, flags);
  ImGui::PopStyleVar(3);

  // the status bar sits below the dockspace: reserve its height plus the item spacing between the two,
  // otherwise the host window is a few pixels taller than the viewport and scrolls
  const float status_height = full_screen_ ? 0.0f : ImGui::GetFrameHeight() + ImGui::GetStyle().ItemSpacing.y;
  const float top_gap = full_screen_ ? 0.0f : ImGui::GetStyle().ItemSpacing.y;
  ImGui::SetCursorPosY(ImGui::GetCursorPosY() + top_gap);
  const ImVec2 dock_size(ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().y - status_height);
  const ImGuiID dock_id = ImGui::GetID("cabana_dockspace");
  if (reset_layout_ || ImGui::DockBuilderGetNode(dock_id) == nullptr) {
    // A shared browser/playback column, with charts and an optional CAN inspector alongside.
    ImGui::DockBuilderRemoveNode(dock_id);
    ImGui::DockBuilderAddNode(dock_id, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodePos(dock_id, ImGui::GetCursorScreenPos());
    ImGui::DockBuilderSetNodeSize(dock_id, dock_size);
    ImGuiID charts = dock_id, left = 0, video = 0, details = 0;
    ImGui::DockBuilderSplitNode(charts, ImGuiDir_Left, 0.28f, &left, &charts);
    const float playback_ratio = hasStream() && can->liveStreaming() ?
      std::clamp((ImGui::GetFrameHeight() * 2 + ImGui::GetStyle().WindowPadding.y * 4) / dock_size.y, 0.1f, 0.5f) : 0.36f;
    ImGui::DockBuilderSplitNode(left, ImGuiDir_Down, playback_ratio, &video, &left);
    ImGui::DockBuilderSplitNode(charts, ImGuiDir_Left, 0.52f, &details, &charts);
    ImGui::DockBuilderDockWindow(MESSAGES_PANEL_ID, left);
    ImGui::DockBuilderDockWindow(LOG_MESSAGES_PANEL, left);
    ImGui::DockBuilderDockWindow(VIDEO_PANEL, video);
    ImGui::DockBuilderDockWindow(CENTER_PANEL, details);
    ImGui::DockBuilderDockWindow(CHARTS_PANEL, charts);
    // All panes resize proportionally. A central node would absorb the entire window
    // resize, squeezing charts while preserving the browser and inspector widths.
    ImGui::DockBuilderGetNode(charts)->LocalFlags &= ~ImGuiDockNodeFlags_CentralNode;
    ImGui::DockBuilderFinish(dock_id);
    reset_layout_ = false;
  }
  // A panel never shrinks past half the width where the signal view's toolbar squishes.
  const float min_panel_width = (SignalView::minimumWidth() + (ImGui::GetStyle().WindowPadding.x + ImGui::GetStyle().WindowBorderSize) * 2) * 0.5f;
  ImGui::PushStyleVar(ImGuiStyleVar_WindowMinSize, ImVec2(min_panel_width, ImGui::GetStyle().WindowMinSize.y));
  ImGui::DockSpace(dock_id, dock_size);
  ImGui::PopStyleVar();
  if (!full_screen_) drawStatusBar();
  ImGui::End();
}

namespace {
// Every workspace panel uses the same docking and floating behavior. Keep the title
// strip visible so moving and closing a panel are always available.
void setNextPanelClass() {
  ImGuiWindowClass window_class;
  window_class.ViewportFlagsOverrideSet = ImGuiViewportFlags_NoAutoMerge;
  window_class.DockNodeFlagsOverrideSet = ImGuiDockNodeFlags_NoWindowMenuButton;
  ImGui::SetNextWindowClass(&window_class);
}

bool beginPanel(const char *name, bool *open, ImGuiWindowFlags flags = 0) {
  ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
  const bool visible = ImGui::Begin(name, open, flags | ImGuiWindowFlags_NoFocusOnAppearing);
  ImGui::PopStyleVar();
  return visible;
}
}  // namespace

void MainWindow::showMessage(const MessageId &id) {
  center_widget_.setMessage(id);
  details_visible_ = true;
  selectPanelTab(CENTER_PANEL);
}

void MainWindow::selectPanelTab(const char *name) {
  // Select a docked panel after it has been submitted, without stealing keyboard
  // focus from the message list (which needs to keep handling the arrow keys).
  nextFrame([name]() {
    if (auto *window = ImGui::FindWindowByName(name); window && window->DockNode && window->DockNode->TabBar) {
      window->DockNode->TabBar->NextSelectedTabId = window->TabId;
    }
  });
}

void MainWindow::drawMessagesPanel() {
  const std::string name = "CAN signals" + std::string(MESSAGES_PANEL_ID);
  setNextPanelClass();
  if (beginPanel(name.c_str(), &messages_visible_) && messages_widget_) {
    help_overlay_.add(messages_widget_->whatsThis(), ImGui::GetCurrentWindow()->Rect());
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextDisabled("%s", messages_widget_->title().c_str());
    ImGui::PopTextWrapPos();
    messages_widget_->draw();
  }
  ImGui::End();
}

void MainWindow::drawLogMessagesPanel() {
  setNextPanelClass();
  if (beginPanel(LOG_MESSAGES_PANEL, &log_messages_visible_) && charts_widget_) {
    help_overlay_.add("<b>openpilot Messages</b><br />Search to filter messages and fields.<br />"
                      "Double-click a field to create a chart.<br />Drag a field onto a chart to compare.",
                      ImGui::GetCurrentWindow()->Rect());
    charts_widget_->drawSignalBrowser();
  }
  ImGui::End();
}

void MainWindow::drawChartsPanel() {
  setNextPanelClass();
  if (beginPanel(CHARTS_PANEL, &charts_visible_, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse)) {
    ImGui::BeginChild("charts", ImVec2(0, 0), ImGuiChildFlags_Borders, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    if (charts_widget_) {
      help_overlay_.add(charts_widget_->whatsThis(), ImGui::GetCurrentWindow()->Rect());
      charts_widget_->draw();
    }
    ImGui::EndChild();
  }
  ImGui::End();
}

void MainWindow::drawVideoPanel() {
  const std::string name = "Playback" + std::string(VIDEO_PANEL);
  setNextPanelClass();
  const bool video_open = beginPanel(name.c_str(), &video_visible_);
  if (!video_dock_title_.empty() && ImGui::IsWindowHovered() && ImGui::IsMouseHoveringRect(
        ImGui::GetCurrentWindow()->TitleBarRect().Min, ImGui::GetCurrentWindow()->TitleBarRect().Max)) {
    ImGui::SetTooltip("%s", video_dock_title_.c_str());
  }
  if (video_widget_) {
    video_widget_->setVisible(video_open);
    if (video_open) {
      const bool live = can->liveStreaming();
      // Live streams only have a toolbar; give it the pane's full content area.
      if (!live) ImGui::BeginChild("video", ImVec2(0, 0), ImGuiChildFlags_Borders);
      help_overlay_.add(video_widget_->whatsThis(), ImGui::GetCurrentWindow()->Rect());
      video_widget_->draw();
      if (!live) ImGui::EndChild();
    }
  }
  ImGui::End();
}

void MainWindow::drawDetailsPanel() {
  setNextPanelClass();
  auto *detail = center_widget_.getDetailWidget();
  const std::string title = detail ? "CAN Details: " + detail->messageId().toString() + "###CenterWidget" : CENTER_PANEL;
  if (beginPanel(title.c_str(), &details_visible_, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse)) {
    ImGui::BeginChild("center", ImVec2(0, 0), ImGuiChildFlags_Borders, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    center_widget_.draw();
    if (detail && help_overlay_.visible()) {
      for (const auto &[text, rect] : detail->helpRects()) help_overlay_.add(text, rect);
    }
    ImGui::EndChild();
  }
  ImGui::End();
}

void MainWindow::draw() {
#ifdef __APPLE__
  full_screen_ = isNativeFullScreen(window_);
#endif
  auto pending = std::move(next_frame_);
  next_frame_.clear();
  for (auto &fn : pending) fn();

  if (ImGui::GetTopMostPopupModal() == nullptr) {
    handleShortcuts();
  } else {
    takeKeyEvents();  // modal dialogs swallow the shortcuts
  }
  if (!full_screen_) drawMenuBar();
  drawDockspace();

  // All workspace panels can be tabbed together, split, floated, or closed.
  if (details_visible_) drawDetailsPanel();
  if (messages_visible_) drawMessagesPanel();
  if (log_messages_visible_) drawLogMessagesPanel();
  if (video_widget_ && !video_visible_) video_widget_->setVisible(false);
  if (video_visible_) drawVideoPanel();
  if (charts_visible_) drawChartsPanel();
  for (auto it = tool_dialogs_.begin(); it != tool_dialogs_.end();) {
    it = (*it)->draw() ? it + 1 : tool_dialogs_.erase(it);
  }

  stream_selector_.draw();
  settings_dialog_.draw();
  drawWaitDialog();
  FileDialog::draw();
  MessageBox::draw();
  help_overlay_.draw();

  // Escape closes the top-most non-modal popup (a menu or a combo list) on its own; the modal dialogs
  // handled Escape themselves above when they were on top
  if (ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
    ImGuiWindow *top = topPopupWindow();
    if (top != nullptr && !(top->Flags & ImGuiWindowFlags_Modal)) ImGui::ClosePopupToLevel(GImGui->OpenPopupStack.Size - 1, true);
  }
}
