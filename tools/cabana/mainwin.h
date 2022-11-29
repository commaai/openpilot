#pragma once

#include <QComboBox>
#include <QDialog>
#include <QJsonDocument>
#include <QMainWindow>
#include <QProgressBar>
#include <QSplitter>
#include <QStackedWidget>
#include <QStatusBar>

#include "selfdrive/ui/qt/widgets/controls.h"
#include "tools/cabana/chartswidget.h"
#include "tools/cabana/detailwidget.h"
#include "tools/cabana/messageswidget.h"
#include "tools/cabana/videowidget.h"

class MainWindow : public QMainWindow {
  Q_OBJECT

public:
  MainWindow();
  void dockCharts(bool dock);
  void showStatusMessage(const QString &msg, int timeout = 0) { statusBar()->showMessage(msg, timeout); }

public slots:
  void loadDBCFromName(const QString &name);
  void loadDBCFromFingerprint();
  void loadDBCFromFile();
  void loadDBCFromClipboard();
  void saveDBCToFile();
  void saveDBCToClipboard();
  void loadRoute(const QString &route = {}, const QString &data_dir = {}, bool use_qcam = false);

signals:
  void showMessage(const QString &msg, int timeout);
  void updateProgressBar(uint64_t cur, uint64_t total, bool success);

protected:
  void createActions();
  void createStatusBar();
  void closeEvent(QCloseEvent *event) override;
  void updateDownloadProgress(uint64_t cur, uint64_t total, bool success);
  void setOption();

  VideoWidget *video_widget;
  MessagesWidget *messages_widget;
  DetailWidget *detail_widget;
  ChartsWidget *charts_widget;
  QSplitter *splitter;
  QWidget *floating_window = nullptr;
  QVBoxLayout *r_layout;
  QProgressBar *progress_bar;
  QLabel *fingerprint_label;
  ElidedLabel *route_label;
  QJsonDocument fingerprint_to_dbc;
  QComboBox *dbc_combo;
};

class LoadRouteDialog : public QDialog {
public:
  LoadRouteDialog(const QString &route, const QString &data_dir, bool use_qcam, QWidget *parent);
  void loadRoute(const QString &route, const QString &data_dir, bool use_qcam);
  QString route_string;

protected:
  void loadClicked();
  void reject() override;

  QLineEdit *route_edit;
  QLabel *loading_label;
  QLabel *title_label;
  QStackedWidget *stacked_layout;
};
