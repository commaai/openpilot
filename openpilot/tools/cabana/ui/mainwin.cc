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
#include "tools/cabana/ui/dialogs/filedialog.h"
#include "tools/cabana/ui/dialogs/messagebox.h"
#include "tools/cabana/ui/imgui_util.h"
#include "tools/cabana/utils/export.h"
#include "tools/cabana/utils/util.h"
#include "tools/replay/py_downloader.h"
#include "tools/replay/util.h"

namespace {
// dock window ids (the visible titles change, the part after ### is the identity)
constexpr const char *MESSAGES_PANEL = "###MessagesPanel";
constexpr const char *VIDEO_PANEL = "###VideoPanel";
constexpr const char *CENTER_PANEL = "###CenterWidget";
constexpr const char *CHARTS_WINDOW = "Charts###ChartsWindow";
}  // namespace

MainWindow::MainWindow(GLFWwindow *window, std::unique_ptr<AbstractStream> stream, const std::string &dbc_file) : window_(window) {
  can = &dummy_;
  loadFingerprints();
  // createActions: the opendbc list is read once, like the Qt menu
  std::error_code ec;
  for (const auto &entry : std::filesystem::directory_iterator(OPENDBC_FILE_PATH, ec)) {
    if (entry.is_regular_file() && entry.path().extension() == ".dbc") {
      opendbc_names_.push_back(entry.path().filename().string());
    }
  }
  std::sort(opendbc_names_.begin(), opendbc_names_.end());
  // createStatusBar
  updateStatus();

  // download handlers are called from download threads
  static auto static_main_win = this;
  installDownloadProgressHandler([](uint64_t cur, uint64_t total, bool success) {
    utils::runOnMainThread([=]() { static_main_win->updateDownloadProgress(cur, total, success); });
  });
  installMessageHandler([](ReplyMsgType type, const std::string msg) {
    utils::runOnMainThread([=]() { static_main_win->showStatusMessage(msg, 2000); });
  });

  connections_.push_back(dbc()->fileChanged.connect([this]() { DBCFileChanged(); }));
  connections_.push_back(UndoStack::instance()->cleanChanged.connect([this](bool clean) {
    window_modified_ = !clean;
    updateWindowTitle();
  }));
  connections_.push_back(settings.changed.connect([this]() { updateStatus(); }));

  nextFrame([this, s = std::shared_ptr<std::unique_ptr<AbstractStream>>(new std::unique_ptr<AbstractStream>(std::move(stream))), dbc_file]() {
    *s ? openStream(std::move(*s), dbc_file) : selectAndOpenStream();
  });
}

void MainWindow::loadFingerprints() {
  std::ifstream json_file((executableDir() / "dbc/car_fingerprint_to_dbc.json"));
  if (!json_file) return;
  const std::string contents{std::istreambuf_iterator<char>(json_file), std::istreambuf_iterator<char>()};
  std::string err;
  auto doc = json11::Json::parse(contents, err);
  if (!err.empty() || !doc.is_object()) return;
  fingerprint_to_dbc_.clear();
  for (const auto &kv : doc.object_items()) {
    if (kv.second.is_string()) {
      fingerprint_to_dbc_.emplace(kv.first, kv.second.string_value());
    }
  }
}

// createActions

void MainWindow::drawFileMenu() {
  const bool has_stream = dynamic_cast<DummyStream *>(can) == nullptr;
  if (ImGui::MenuItem("Open Stream...")) selectAndOpenStream();
  if (ImGui::MenuItem("Close stream", nullptr, false, has_stream)) closeStream();
  if (ImGui::MenuItem("Export to CSV...", nullptr, false, has_stream)) exportToCSV();
  ImGui::Separator();

  if (ImGui::MenuItem("New DBC File", "Ctrl+N")) newFile();
  if (ImGui::MenuItem("Open DBC File...", "Ctrl+O")) openFile();

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
  if (ImGui::MenuItem("Load DBC From Clipboard")) loadFromClipboard();

  ImGui::Separator();
  const int cnt = dbc()->nonEmptyDBCCount();
  const std::string save_text = cnt > 1 ? "Save " + std::to_string(cnt) + " DBCs..." : "Save DBC...";
  if (ImGui::MenuItem(save_text.c_str(), "Ctrl+S", false, cnt > 0)) save();
  if (ImGui::MenuItem("Save DBC As...", "Ctrl+Shift+S", false, cnt == 1)) saveAs();
  // TODO: Support clipboard for multiple files
  if (ImGui::MenuItem("Copy DBC To Clipboard", nullptr, false, cnt == 1)) saveToClipboard();

  ImGui::Separator();
  if (ImGui::MenuItem("Settings...")) setOption();

  ImGui::Separator();
  if (ImGui::MenuItem("Exit", "Ctrl+Q")) close();
}

void MainWindow::drawMenuBar() {
  if (!ImGui::BeginMainMenuBar()) return;
  if (ImGui::BeginMenu("File")) {
    drawFileMenu();
    ImGui::EndMenu();
  }

  // Edit Menu
  if (ImGui::BeginMenu("Edit")) {
    auto stack = UndoStack::instance();
    const std::string undo_text = stack->canUndo() ? "Undo " + stack->undoText() : "Undo";
    const std::string redo_text = stack->canRedo() ? "Redo " + stack->redoText() : "Redo";
    if (ImGui::MenuItem(undo_text.c_str(), "Ctrl+Z", false, stack->canUndo())) stack->undo();
    if (ImGui::MenuItem(redo_text.c_str(), "Ctrl+Shift+Z", false, stack->canRedo())) stack->redo();
    ImGui::EndMenu();
  }

  // View Menu
  if (ImGui::BeginMenu("View")) {
    if (ImGui::MenuItem("Full Screen", "F11", full_screen_)) toggleFullScreen();
    ImGui::Separator();
    ImGui::MenuItem(messages_widget_ ? messages_widget_->title().c_str() : "MESSAGES", nullptr, &messages_visible_);
    ImGui::MenuItem(video_dock_title_.empty() ? "##video_dock" : video_dock_title_.c_str(), nullptr, &video_visible_);
    ImGui::Separator();
    if (ImGui::MenuItem("Reset Window Layout")) {
      messages_visible_ = video_visible_ = true;
      reset_layout_ = true;
    }
    ImGui::EndMenu();
  }

  // Tools Menu
  const bool has_stream = dynamic_cast<DummyStream *>(can) == nullptr;
  if (ImGui::BeginMenu("Tools", has_stream)) {
    if (ImGui::MenuItem("Find Similar Bits")) findSimilarBits();
    if (ImGui::MenuItem("Find Signal")) findSignal();
    ImGui::EndMenu();
  }

  // Help Menu
  if (ImGui::BeginMenu("Help")) {
    if (ImGui::MenuItem("Help", "F1")) onlineHelp();
    ImGui::EndMenu();
  }
  ImGui::EndMainMenuBar();
}

void MainWindow::createDockWidgets() {
  widget_connections_.clear();
  messages_widget_ = std::make_unique<MessagesWidget>();
  widget_connections_.push_back(messages_widget_->msgSelectionChanged.connect([this](const MessageId &id) { center_widget_.setMessage(id); }));

  // right panel
  charts_widget_ = std::make_unique<ChartsWidget>();
  center_widget_.setChartsWidget(charts_widget_.get());  // Qt's DetailWidget reaches the charts through its MainWindow parent
  video_widget_ = std::make_unique<VideoWidget>();
  widget_connections_.push_back(charts_widget_->toggleChartsDocking.connect([this]() { toggleChartsDocking(); }));
  widget_connections_.push_back(charts_widget_->showTip.connect([this](double sec) { video_widget_->showThumbnail(sec); }));
}

void MainWindow::updateStatus() {
  status_label_ = "Cached Minutes:" + std::to_string(settings.max_cached_minutes) + " FPS:" + std::to_string(settings.fps);
}

void MainWindow::showStatusMessage(const std::string &msg, int timeout_ms) {
  status_message_ = msg;
  status_message_until_ = timeout_ms > 0 ? ImGui::GetTime() + timeout_ms / 1000.0 : 0;
}

void MainWindow::updateWindowTitle() {
  // setWindowFilePath + setWindowModified
  std::string title;
  for (auto f : dbc()->allDBCFiles()) {
    if (!title.empty()) title += " | ";
    title += "(" + toString(dbc()->sources(f)) + ") " + f->name();
  }
  if (window_modified_) title += "*";
  if (!title.empty()) title += " - ";
  title += "Cabana";
  glfwSetWindowTitle(window_, title.c_str());
}

void MainWindow::DBCFileChanged() {
  UndoStack::instance()->clear();
  // the file menu state (save texts, enabled flags) is derived from dbc() when the menu is drawn
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

void MainWindow::closeStream() {
  openStream(std::make_unique<DummyStream>());
  if (dbc()->nonEmptyDBCCount() > 0) {
    dbc()->fileChanged();
  }
  showStatusMessage("stream closed");
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
        showStatusMessage("DBC File " + fn + " loaded", 2000);
      } else {
        MessageBox::warning("Failed to load DBC file", "Failed to parse DBC file " + fn, error);
      }
      if (then) then();
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
    MessageBox::warning("Load From Clipboard", "No clipboard tool found. Install xclip (X11) or wl-clipboard (Wayland).");
    return;
  }
  if (text.empty()) {
    MessageBox::warning("Load From Clipboard", "Clipboard is empty.");
    return;
  }

  closeFile(s, [s, text]() {
    std::string error;
    bool ret = dbc()->open(s, std::string(""), text, &error);
    if (ret && dbc()->nonEmptyDBCCount() > 0) {
      MessageBox::information("Load From Clipboard", "DBC Successfully Loaded!");
    } else {
      MessageBox::warning("Failed to load DBC from clipboard", "Make sure that you paste the text with correct format.", error);
    }
  });
}

// stream threads read the global `can` until its destructor joins them
MainWindow::~MainWindow() {
  installDownloadProgressHandler(nullptr);
  installMessageHandler(nullptr);
  widget_connections_.clear();
  stream_connections_.clear();
  stream_.reset();
  can = nullptr;
}

void MainWindow::openStream(std::unique_ptr<AbstractStream> stream, const std::string &dbc_file) {
  stream_connections_.clear();
  wait_dlg_connection_.disconnect();
  wait_dlg_open_ = false;
  can = &dummy_;
  stream_.reset();
  startStream(std::move(stream), dbc_file);
}

void MainWindow::startStream(std::unique_ptr<AbstractStream> stream, const std::string &dbc_file) {
  center_widget_.clear();
  widget_connections_.clear();
  messages_widget_.reset();
  video_widget_.reset();
  charts_widget_.reset();

  stream_ = std::move(stream);  // take ownership
  can = stream_.get();
  stream_connections_.push_back(can->error.connect([](const std::string &msg) {
    MessageBox::warning("Error", msg);
  }));
  can->start();

  loadFile(dbc_file, SOURCE_ALL, [this]() {
    showStatusMessage("Stream [" + can->routeName() + "] started", 2000);

    bool has_stream = dynamic_cast<DummyStream *>(can) == nullptr;
    createDockWidgets();

    video_dock_title_ = can->routeName();
    // Don't overwrite already loaded DBC
    if (!dbc()->nonEmptyDBCCount()) {
      newFile();
    }

    stream_connections_.push_back(can->eventsMerged.connect([this](const MessageEventsMap &) { eventsMerged(); }));

    if (has_stream) {
      wait_dlg_text_ = can->liveStreaming() ? "Waiting for the live stream to start..." : "Loading segment data...";
      wait_dlg_value_ = 0;
      wait_dlg_open_ = true;
      wait_dlg_connection_ = can->eventsMerged.connect([this](const MessageEventsMap &) {
        wait_dlg_open_ = false;
        wait_dlg_connection_.disconnect();
      });
    }
  });
}

void MainWindow::eventsMerged() {
  if (!can->liveStreaming() && std::exchange(car_fingerprint_, can->carFingerprint()) != car_fingerprint_) {
    video_dock_title_ = "ROUTE: " + can->routeName() + "  FINGERPRINT: " + (car_fingerprint_.empty() ? "Unknown Car" : car_fingerprint_);
    // Don't overwrite already loaded DBC
    auto it = fingerprint_to_dbc_.find(car_fingerprint_);
    if (!dbc()->nonEmptyDBCCount() && it != fingerprint_to_dbc_.end()) {
      nextFrame([this, dbc_name = it->second]() { loadDBCFromOpendbc(dbc_name + ".dbc"); });
    }
  }
}

void MainWindow::save(std::function<void()> then) {
  // Save all open DBC files
  std::vector<DBCFile *> files;
  for (auto dbc_file : dbc()->allDBCFiles()) {
    if (dbc_file->isEmpty()) continue;
    files.push_back(dbc_file);
  }
  auto next = std::make_shared<std::function<void(size_t)>>();
  *next = [this, files, next, then](size_t i) {
    if (i >= files.size()) {
      if (then) then();
      return;
    }
    saveFile(files[i], [next, i]() { (*next)(i + 1); });
  };
  (*next)(0);
}

void MainWindow::saveAs(std::function<void()> then) {
  // Save as all open DBC files. Should not be called with more than 1 file open
  std::vector<DBCFile *> files;
  for (auto dbc_file : dbc()->allDBCFiles()) {
    if (dbc_file->isEmpty()) continue;
    files.push_back(dbc_file);
  }
  auto next = std::make_shared<std::function<void(size_t)>>();
  *next = [this, files, next, then](size_t i) {
    if (i >= files.size()) {
      if (then) then();
      return;
    }
    saveFileAs(files[i], [next, i]() { (*next)(i + 1); });
  };
  (*next)(0);
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
  // Copy all open DBC files to clipboard. Should not be called with more than 1 file open
  for (auto dbc_file : dbc()->allDBCFiles()) {
    if (dbc_file->isEmpty()) continue;
    saveFileToClipboard(dbc_file);
  }
}

void MainWindow::saveFileToClipboard(DBCFile *dbc_file) {
  assert(dbc_file != nullptr);
  if (utils::setClipboardText(dbc_file->generateDBC())) {
    MessageBox::information("Copy To Clipboard", "DBC Successfully copied!");
  } else {
    MessageBox::warning("Copy To Clipboard", "Failed to copy DBC to clipboard. Install xclip (X11) or wl-clipboard (Wayland).");
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
      if (ImGui::MenuItem("New DBC File...")) newFile(ss);
      if (ImGui::MenuItem("Open DBC File...")) openFile(ss);
      if (ImGui::MenuItem("Load DBC From Clipboard...")) loadFromClipboard(ss, false);

      // Show sub-menu for each dbc for this source.
      if (dbc_file) {
        ImGui::Separator();
        ImGui::MenuItem((dbc_file->name() + " (" + toString(dbc()->sources(dbc_file)) + ")").c_str(), nullptr, false, false);
        if (ImGui::MenuItem("Save...")) saveFile(dbc_file);
        if (ImGui::MenuItem("Save As...")) saveFileAs(dbc_file);
        if (ImGui::MenuItem("Copy to Clipboard...")) saveFileToClipboard(dbc_file);
        if (ImGui::MenuItem("Remove from this bus...")) closeFile(ss, {});
        if (ImGui::MenuItem("Remove from all buses...")) closeFile(dbc_file);
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
  std::string text = "You have unsaved changes. Press ok to save them, cancel to discard.";
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
  if (wait_dlg_open_) wait_dlg_value_ = (int)((cur / (double)total) * 100);
  if (success && cur < total) {
    progress_value_ = (cur / (double)total);
    progress_text_ = "Downloading " + std::to_string((int)(progress_value_ * 100)) + "% (" + formattedDataSize(total) + ")";
    progress_visible_ = true;
  } else {
    progress_visible_ = false;
  }
}

void MainWindow::toggleChartsDocking() {
  charts_floating_ = !charts_floating_;
  charts_widget_->setIsDocked(!charts_floating_);
}

void MainWindow::close() {
  if (closing_) return;
  closing_ = true;
  remindSaveChanges([this]() { finishClose(); });
}

void MainWindow::finishClose() {
  installDownloadProgressHandler(nullptr);
  installMessageHandler(nullptr);

  // save states
  if (messages_widget_) {
    settings.message_header_state = messages_widget_->saveHeaderState();
  }

  saveSessionState();
  settings.save();
  exited_ = true;
}

void MainWindow::setOption() {
  settings_dialog_.open();
}

void MainWindow::findSimilarBits() {
  auto dlg = std::make_unique<FindSimilarBitsDlg>();
  widget_connections_.push_back(dlg->openMessage.connect([this](const MessageId &id) { messages_widget_->selectMessage(id); }));
  tool_dialogs_.push_back(std::move(dlg));
}

void MainWindow::findSignal() {
  auto dlg = std::make_unique<FindSignalDlg>();
  widget_connections_.push_back(dlg->openMessage.connect([this](const MessageId &id) { messages_widget_->selectMessage(id); }));
  tool_dialogs_.push_back(std::move(dlg));
}

void MainWindow::onlineHelp() {
  help_overlay_ = !help_overlay_;
}

void MainWindow::toggleFullScreen() {
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
}

void MainWindow::saveSessionState() {
  settings.recent_dbc_file = "";
  settings.active_msg_id = "";
  settings.selected_msg_ids.clear();
  settings.active_charts.clear();

  for (auto &f : dbc()->allDBCFiles())
    if (!f->isEmpty()) { settings.recent_dbc_file = f->filename; break; }

  if (auto *detail = center_widget_.getDetailWidget()) {
    auto [active_id, ids] = detail->serializeMessageIds();
    settings.active_msg_id = active_id;
    settings.selected_msg_ids = ids;
  }
  if (charts_widget_) {
    settings.active_charts = charts_widget_->serializeChartIds();
  }
}

void MainWindow::restoreSessionState() {
  if (settings.recent_dbc_file.empty() || dbc()->nonEmptyDBCCount() == 0) return;

  std::string dbc_file;
  for (auto &f : dbc()->allDBCFiles())
    if (!f->isEmpty()) { dbc_file = f->filename; break; }
  if (dbc_file != settings.recent_dbc_file) return;

  if (!settings.selected_msg_ids.empty()) {
    center_widget_.ensureDetailWidget()->restoreTabs(settings.active_msg_id, settings.selected_msg_ids);
  }

  if (charts_widget_ != nullptr && !settings.active_charts.empty()) {
    charts_widget_->restoreChartsFromIds(settings.active_charts);
  }
}

// createShortcuts + QKeySequence menu shortcuts

void MainWindow::handleShortcuts() {
  const ImGuiIO &io = ImGui::GetIO();
  const bool ctrl = io.KeyCtrl || io.KeySuper;
  const bool shift = io.KeyShift;
  // a focused line edit consumes Space (ShortcutOverride) but not the Ctrl/F-key sequences
  if (ImGui::IsKeyPressed(ImGuiKey_Space, false) && can && !io.WantTextInput) can->pause(!can->isPaused());
  if (ImGui::IsKeyPressed(ImGuiKey_F1, false)) onlineHelp();
  if (ImGui::IsKeyPressed(ImGuiKey_F11, false)) toggleFullScreen();
  if (!ctrl) return;
  if (ImGui::IsKeyPressed(ImGuiKey_N, false)) newFile();
  if (ImGui::IsKeyPressed(ImGuiKey_O, false)) openFile();
  if (ImGui::IsKeyPressed(ImGuiKey_S, false)) {
    if (shift) {
      if (dbc()->nonEmptyDBCCount() == 1) saveAs();
    } else if (dbc()->nonEmptyDBCCount() > 0) {
      save();
    }
  }
  if (ImGui::IsKeyPressed(ImGuiKey_Z, false)) shift ? UndoStack::instance()->redo() : UndoStack::instance()->undo();
  if (ImGui::IsKeyPressed(ImGuiKey_Y, false)) UndoStack::instance()->redo();
  if (ImGui::IsKeyPressed(ImGuiKey_Q, false)) close();
}

void MainWindow::drawStatusBar() {
  ImGui::PushStyleColor(ImGuiCol_ChildBg, ImGui::GetStyle().Colors[ImGuiCol_MenuBarBg]);
  ImGui::BeginChild("status_bar", ImVec2(0, ImGui::GetFrameHeight()), ImGuiChildFlags_None, ImGuiWindowFlags_NoScrollbar);
  ImGui::AlignTextToFramePadding();
  // a temporary message hides the normal widgets, permanent widgets stay on the right
  if (!status_message_.empty() && (status_message_until_ == 0 || ImGui::GetTime() < status_message_until_)) {
    ImGui::TextUnformatted(status_message_.c_str());
  } else {
    status_message_.clear();
    ImGui::TextUnformatted("For Help, Press F1");
  }
  float right = ImGui::GetContentRegionAvail().x - ImGui::CalcTextSize(status_label_.c_str()).x;
  if (progress_visible_) {
    ImGui::SameLine(right - 300.0f - ImGui::GetStyle().ItemSpacing.x);
    ImGui::ProgressBar(progress_value_, ImVec2(300.0f, 16.0f), progress_text_.c_str());
  }
  ImGui::SameLine(right);
  ImGui::TextUnformatted(status_label_.c_str());
  ImGui::EndChild();
  ImGui::PopStyleColor();
}

void MainWindow::drawWaitDialog() {
  const char *id = "###WaitDialog";
  if (wait_dlg_open_ && !ImGui::IsPopupOpen(id)) ImGui::OpenPopup(id);
  if (!ImGui::IsPopupOpen(id)) return;  // keep submitting until CloseCurrentPopup ran, a stale modal blocks all input
  ImGui::SetNextWindowSize(ImVec2(400.0f, 0.0f));
  ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
  if (ImGui::BeginPopupModal(id, nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::TextUnformatted(wait_dlg_text_.c_str());
    ImGui::ProgressBar(wait_dlg_value_ / 100.0f, ImVec2(-1.0f, 0.0f));
    if (ImGui::Button("Abort")) {
      wait_dlg_open_ = false;
      close();
    }
    if (!wait_dlg_open_) ImGui::CloseCurrentPopup();
    ImGui::EndPopup();
  }
}

// HelpOverlay: dims the window and shows each widget's whatsThis text at its center; any click closes it
void MainWindow::drawHelpOverlay() {
  if (!help_overlay_) return;
  const ImGuiViewport *viewport = ImGui::GetMainViewport();
  ImDrawList *dl = ImGui::GetForegroundDrawList();
  dl->AddRectFilled(viewport->Pos, ImVec2(viewport->Pos.x + viewport->Size.x, viewport->Pos.y + viewport->Size.y), IM_COL32(0, 0, 0, 50));
  for (const auto &[text, rect] : help_texts_) {
    if (text.empty()) continue;
    const ImVec2 center((rect.Min.x + rect.Max.x) * 0.5f, (rect.Min.y + rect.Max.y) * 0.5f);
    const ImVec2 size = ImGui::CalcTextSize(text.c_str(), nullptr, false, 400.0f);
    const ImVec2 min(center.x - size.x * 0.5f - 8.0f, center.y - size.y * 0.5f - 8.0f);
    const ImVec2 max(center.x + size.x * 0.5f + 8.0f, center.y + size.y * 0.5f + 8.0f);
    dl->AddRectFilled(min, max, ImGui::GetColorU32(ImGuiCol_PopupBg));
    dl->AddText(nullptr, 0.0f, ImVec2(min.x + 8.0f, min.y + 8.0f), ImGui::GetColorU32(ImGuiCol_Text), text.c_str(), nullptr, 400.0f);
  }
  help_texts_.clear();
  if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) help_overlay_ = false;
}

void MainWindow::drawDockspace() {
  const ImGuiViewport *viewport = ImGui::GetMainViewport();
  ImGui::SetNextWindowPos(viewport->WorkPos);
  ImGui::SetNextWindowSize(viewport->WorkSize);
  ImGui::SetNextWindowViewport(viewport->ID);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
  const ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize |
                                 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus |
                                 ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoBackground;
  ImGui::Begin("##host", nullptr, flags);
  ImGui::PopStyleVar(3);

  const float status_height = full_screen_ ? 0.0f : ImGui::GetFrameHeight();
  const ImVec2 dock_size(ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().y - status_height);
  const ImGuiID dock_id = ImGui::GetID("cabana_dockspace");
  if (reset_layout_) {
    // createDockWindows: messages left, video (with charts) right, center widget in the middle
    ImGui::DockBuilderRemoveNode(dock_id);
    ImGui::DockBuilderAddNode(dock_id, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(dock_id, dock_size);
    ImGuiID center = dock_id, left = 0, right = 0;
    ImGui::DockBuilderSplitNode(center, ImGuiDir_Left, 0.28f, &left, &center);
    ImGui::DockBuilderSplitNode(center, ImGuiDir_Right, 0.4f, &right, &center);
    ImGui::DockBuilderDockWindow(MESSAGES_PANEL, left);
    ImGui::DockBuilderDockWindow(VIDEO_PANEL, right);
    ImGui::DockBuilderDockWindow(CENTER_PANEL, center);
    ImGui::DockBuilderGetNode(center)->LocalFlags |= ImGuiDockNodeFlags_NoTabBar;
    ImGui::DockBuilderFinish(dock_id);
    reset_layout_ = false;
  }
  ImGui::DockSpace(dock_id, dock_size);
  if (!full_screen_) drawStatusBar();
  ImGui::End();
}

void MainWindow::draw() {
  auto pending = std::move(next_frame_);
  next_frame_.clear();
  for (auto &fn : pending) fn();

  if (!MessageBox::isOpen() && !FileDialog::isOpen() && !stream_selector_.isOpen()) handleShortcuts();
  if (!full_screen_) drawMenuBar();
  drawDockspace();

  if (ImGui::Begin(CENTER_PANEL)) center_widget_.draw();
  ImGui::End();
  if (messages_widget_ && messages_visible_) {
    const std::string name = messages_widget_->title() + MESSAGES_PANEL;
    if (ImGui::Begin(name.c_str(), &messages_visible_)) {
      help_texts_.emplace_back(messages_widget_->whatsThis(), ImGui::GetCurrentWindow()->Rect());
      messages_widget_->draw();
    }
    ImGui::End();
  }
  if (video_widget_ && video_visible_) {
    const std::string name = video_dock_title_ + VIDEO_PANEL;
    if (ImGui::Begin(name.c_str(), &video_visible_)) {
      // splitter between video and charts
      const ImVec2 avail = ImGui::GetContentRegionAvail();
      const bool live = can->liveStreaming();
      float video_h = charts_floating_ ? avail.y : std::clamp(avail.y * video_splitter_ratio_, 1.0f, avail.y - 1.0f);
      if (live) video_h = 1.0f;  // display video at minimum size.
      ImGui::BeginChild("video", ImVec2(0, video_h));
      help_texts_.emplace_back(video_widget_->whatsThis(), ImGui::GetCurrentWindow()->Rect());
      video_widget_->draw();
      ImGui::EndChild();
      if (!charts_floating_) {
        ImGui::InvisibleButton("##splitter", ImVec2(-1.0f, 6.0f));
        if (ImGui::IsItemActive() && !live) video_splitter_ratio_ = std::clamp((ImGui::GetMousePos().y - ImGui::GetWindowPos().y) / avail.y, 0.0f, 1.0f);
        if (ImGui::IsItemHovered() && !live) ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
        ImGui::BeginChild("charts", ImVec2(0, 0));
        help_texts_.emplace_back(charts_widget_->whatsThis(), ImGui::GetCurrentWindow()->Rect());
        charts_widget_->draw();
        ImGui::EndChild();
      }
    }
    ImGui::End();
  }
  if (charts_widget_ && charts_floating_) {
    bool open = true;
    ImGui::SetNextWindowSize(ImGui::GetMainViewport()->WorkSize, ImGuiCond_Appearing);
    if (ImGui::Begin(CHARTS_WINDOW, &open)) charts_widget_->draw();
    ImGui::End();
    if (!open) toggleChartsDocking();
  }
  for (auto it = tool_dialogs_.begin(); it != tool_dialogs_.end();) {
    it = (*it)->draw() ? it + 1 : tool_dialogs_.erase(it);
  }

  stream_selector_.draw();
  settings_dialog_.draw();
  drawWaitDialog();
  FileDialog::draw();
  MessageBox::draw();
  drawHelpOverlay();
}
