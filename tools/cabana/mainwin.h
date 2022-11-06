#pragma once

#include <QComboBox>
#include <QDialog>
#include <QJsonDocument>
#include <QProgressBar>
#include <QSplitter>
#include <QStatusBar>
#include <QTextEdit>

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
  void saveDBC();

signals:
  void showMessage(const QString &msg, int timeout);
  void updateProgressBar(uint64_t cur, uint64_t total, bool success);

protected:
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
  QStatusBar *status_bar;
  QJsonDocument fingerprint_to_dbc;
  QComboBox *dbc_combo;
};


class LoadDBCDialog : public QDialog {
  Q_OBJECT

public:
  LoadDBCDialog(QWidget *parent);
  QTextEdit *dbc_edit;
};

class SaveDBCDialog : public QDialog {
  Q_OBJECT

public:
  SaveDBCDialog(QWidget *parent);
  void copytoClipboard();
  void saveAs();
  QTextEdit *dbc_edit;
};
