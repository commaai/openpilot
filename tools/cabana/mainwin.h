#pragma once

#include <QComboBox>
#include <QDialog>
#include <QJsonDocument>
#include <QProgressBar>
#include <QStatusBar>
#include <QTextEdit>
#include <QToolButton>

#include "tools/cabana/chartswidget.h"
#include "tools/cabana/detailwidget.h"
#include "tools/cabana/messageswidget.h"
#include "tools/cabana/videowidget.h"

class MainWindow : public QWidget {
  Q_OBJECT

public:
  MainWindow();
  void dockCharts(bool dock);
  void showStatusMessage(const QString &msg, int timeout = 0) { status_bar->showMessage(msg, timeout); }

public slots:
  void loadDBCFromName(const QString &name);
  void loadDBCFromFingerprint();
  void loadDBCFromPaste();
  void loadRoute(const QString &route, const QString &data_dir, bool use_qcam = false);

signals:
  void logMessageFromReplay(const QString &msg, int timeout);
  void updateProgressBar(uint64_t cur, uint64_t total, bool success);

protected:
  QToolButton *initRouteControl();
  void closeEvent(QCloseEvent *event) override;
  void updateDownloadProgress(uint64_t cur, uint64_t total, bool success);
  void setOption();
  // void openRouteDialog();

  VideoWidget *video_widget;
  MessagesWidget *messages_widget;
  DetailWidget *detail_widget;
  ChartsWidget *charts_widget;
  QWidget *floating_window = nullptr;
  QVBoxLayout *r_layout;
  QProgressBar *progress_bar;
  QStatusBar *status_bar;
  QToolButton *route_btn;
  QJsonDocument fingerprint_to_dbc;
  QLabel *fingerprint_label;
  QComboBox *dbc_combo;

  CANMessages can_message;
};

class LoadDBCDialog : public QDialog {
  Q_OBJECT

public:
  LoadDBCDialog(QWidget *parent);
  QTextEdit *dbc_edit;
};

class LoadRouteDialog : public QDialog {
  Q_OBJECT

public:
  LoadRouteDialog(QWidget *parent);
};
