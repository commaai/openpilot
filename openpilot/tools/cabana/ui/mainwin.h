#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "imgui_internal.h"
#include "tools/cabana/dbc/dbcmanager.h"
#include "tools/cabana/streams/abstractstream.h"
#include "tools/cabana/ui/dialogs/settingsdialog.h"
#include "tools/cabana/ui/dialogs/streamselector.h"
#include "tools/cabana/ui/tools/findsignal.h"
#include "tools/cabana/ui/tools/findsimilarbits.h"
#include "tools/cabana/ui/tools/tooldialog.h"
#include "tools/cabana/ui/chart/chartswidget.h"
#include "tools/cabana/ui/widgets/detailwidget.h"
#include "tools/cabana/ui/widgets/messageswidget.h"
#include "tools/cabana/ui/widgets/videowidget.h"

struct GLFWwindow;

class MainWindow {
public:
  MainWindow(GLFWwindow *window, std::unique_ptr<AbstractStream> stream, const std::string &dbc_file);
  ~MainWindow();
  void draw();
  void toggleChartsDocking();
  void close();  // remind unsaved changes, save state, exit
  bool exited() const { return exited_; }
  void showStatusMessage(const std::string &msg, int timeout_ms = 0);
  void loadFile(const std::string &fn, SourceSet s = SOURCE_ALL, std::function<void()> then = {});

  void selectAndOpenStream();
  void openStream(std::unique_ptr<AbstractStream> stream, const std::string &dbc_file = {});
  void closeStream();
  void exportToCSV();

  void newFile(SourceSet s = SOURCE_ALL);
  void openFile(SourceSet s = SOURCE_ALL);
  void loadDBCFromOpendbc(const std::string &name);
  void save(std::function<void()> then = {});
  void saveAs(std::function<void()> then = {});
  void saveToClipboard();

private:
  void startStream(std::unique_ptr<AbstractStream> stream, const std::string &dbc_file);
  void remindSaveChanges(std::function<void()> then);
  void closeFile(SourceSet s, std::function<void()> then);
  void closeFile(DBCFile *dbc_file);
  void saveFiles(bool as, std::function<void()> then);
  void saveFile(DBCFile *dbc_file, std::function<void()> then = {});
  void saveFileAs(DBCFile *dbc_file, std::function<void()> then = {});
  void saveFileToClipboard(DBCFile *dbc_file);
  void loadFingerprints();
  void loadFromClipboard(SourceSet s = SOURCE_ALL, bool close_all = true);
  void updateRecentFiles(const std::string &fn);
  void DBCFileChanged();
  void updateDownloadProgress(uint64_t cur, uint64_t total, bool success);
  void setOption();
  void findSimilarBits();
  void findSignal();
  void onlineHelp();
  void toggleFullScreen();
  void updateWindowTitle();
  void eventsMerged();
  void saveSessionState();
  void restoreSessionState();
  void finishClose();
  void nextFrame(std::function<void()> fn) { next_frame_.push_back(std::move(fn)); }
  void createDockWidgets();

  void handleShortcuts();
  void drawMenuBar();
  void drawFileMenu();
  void drawManageDBCsMenu();
  void drawRecentFilesMenu();
  void drawDockspace();
  void drawStatusBar();
  void drawWaitDialog();
  void drawHelpOverlay();

  GLFWwindow *window_;
  std::unique_ptr<AbstractStream> startup_stream_;  // opened on the first frame
  std::unique_ptr<AbstractStream> stream_;  // `can` points here, or at dummy_ when no stream is open
  DummyStream dummy_;
  std::unique_ptr<MessagesWidget> messages_widget_;
  CenterWidget center_widget_;
  std::unique_ptr<VideoWidget> video_widget_;
  std::unique_ptr<ChartsWidget> charts_widget_;
  StreamSelector stream_selector_;
  SettingsDialog settings_dialog_;
  std::unordered_map<std::string, std::string> fingerprint_to_dbc_;
  std::vector<std::string> opendbc_names_;
  enum { MAX_RECENT_FILES = 15 };
  std::string car_fingerprint_;
  std::string video_dock_title_;
  bool messages_visible_ = true;
  bool video_visible_ = true;
  bool reset_layout_ = false;
  bool full_screen_ = false;
#ifndef __APPLE__
  int windowed_rect_[4] = {0, 0, 1600, 900};
#endif
  bool charts_floating_ = false;
  float video_splitter_ratio_ = -1.0f;  // < 0: the video widget is at its size hint
  std::vector<std::pair<std::string, ImRect>> help_texts_;
  std::vector<std::unique_ptr<ToolDialog>> tool_dialogs_;
  bool help_overlay_ = false;
  int help_overlay_frame_ = -1;
  bool closing_ = false;
  bool exited_ = false;
  bool window_modified_ = false;
  // status bar
  std::string status_message_;
  double status_message_until_ = 0;
  bool progress_visible_ = false;
  float progress_value_ = 0;
  std::string progress_text_;
  // "Loading segment data..." dialog
  bool wait_dlg_open_ = false;
  double wait_dlg_show_at_ = 0;
  bool manage_dbcs_enabled_ = false;
  std::string wait_dlg_text_;
  int wait_dlg_value_ = 0;
  Connection wait_dlg_connection_;
  std::vector<std::function<void()>> next_frame_;
  Connections connections_;
  Connections stream_connections_;
  Connections widget_connections_;
};
