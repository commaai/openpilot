#pragma once

#include <QDockWidget>
#include <QJsonDocument>
#include <QMainWindow>
#include <QMenu>
#include <QProgressBar>
#include <QSplitter>
#include <QStatusBar>
#include <set>

#include "tools/cabana/chart/chartswidget.h"
#include "tools/cabana/dbc/dbcmanager.h"
#include "tools/cabana/detailwidget.h"
#include "tools/cabana/messageswidget.h"
#include "tools/cabana/videowidget.h"
#include "tools/cabana/tools/findsimilarbits.h"

class MainWindow : public QMainWindow {
  Q_OBJECT

public:
  MainWindow();
  void dockCharts(bool dock);
  void showStatusMessage(const QString &msg, int timeout = 0) { statusBar()->showMessage(msg, timeout); }
  void loadFile(const QString &fn, SourceSet s = SOURCE_ALL);
  ChartsWidget *charts_widget = nullptr;

public slots:
  void openStream();
  void closeStream();
  void exportToCSV();
  void changingStream();
  void streamStarted();

  void newFile(SourceSet s = SOURCE_ALL);
  void openFile(SourceSet s = SOURCE_ALL);
  void openRecentFile();
  void loadDBCFromOpendbc(const QString &name);
  void save();
  void saveAs();
  void saveToClipboard();

signals:
  void showMessage(const QString &msg, int timeout);
  void updateProgressBar(uint64_t cur, uint64_t total, bool success);

protected:
  void remindSaveChanges();
  void closeFile(SourceSet s = SOURCE_ALL);
  void closeFile(DBCFile *dbc_file);
  void saveFile(DBCFile *dbc_file);
  void saveFileAs(DBCFile *dbc_file);
  void saveFileToClipboard(DBCFile *dbc_file);
  void loadFingerprints();
  void loadFromClipboard(SourceSet s = SOURCE_ALL, bool close_all = true);
  void autoSave();
  void cleanupAutoSaveFile();
  void updateRecentFiles(const QString &fn);
  void updateRecentFileActions();
  void createActions();
  void createDockWindows();
  void createStatusBar();
  void createShortcuts();
  void closeEvent(QCloseEvent *event) override;
  void DBCFileChanged();
  void updateDownloadProgress(uint64_t cur, uint64_t total, bool success);
  void setOption();
  void findSimilarBits();
  void findSignal();
  void undoStackCleanChanged(bool clean);
  void undoStackIndexChanged(int index);
  void onlineHelp();
  void toggleFullScreen();
  void updateStatus();
  void updateLoadSaveMenus();
  void createDockWidgets();
  void eventsMerged();

  VideoWidget *video_widget = nullptr;
  QDockWidget *video_dock;
  QDockWidget *messages_dock;
  MessagesWidget *messages_widget = nullptr;
  CenterWidget *center_widget;
  QWidget *floating_window = nullptr;
  QVBoxLayout *charts_layout;
  QProgressBar *progress_bar;
  QLabel *status_label;
  QJsonDocument fingerprint_to_dbc;
  QStringList opendbc_names;
  QSplitter *video_splitter = nullptr;
  enum { MAX_RECENT_FILES = 15 };
  QAction *recent_files_acts[MAX_RECENT_FILES] = {};
  QMenu *open_recent_menu = nullptr;
  QMenu *manage_dbcs_menu = nullptr;
  QMenu *tools_menu = nullptr;
  QAction *close_stream_act = nullptr;
  QAction *export_to_csv_act = nullptr;
  QAction *save_dbc = nullptr;
  QAction *save_dbc_as = nullptr;
  QAction *copy_dbc_to_clipboard = nullptr;
  QString car_fingerprint;
  int prev_undostack_index = 0;
  int prev_undostack_count = 0;
  QByteArray default_state;
};

class HelpOverlay : public QWidget {
  Q_OBJECT
public:
  HelpOverlay(MainWindow *parent);

protected:
  void drawHelpForWidget(QPainter &painter, QWidget *w);
  void paintEvent(QPaintEvent *event) override;
  void mouseReleaseEvent(QMouseEvent *event) override;
  bool eventFilter(QObject *obj, QEvent *event) override;
};
