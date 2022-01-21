#pragma once

#include <curl/curl.h>

#include <QProgressBar>
#include <QStackedWidget>
#include <QString>
#include <QWidget>

class Setup : public QStackedWidget {
  Q_OBJECT

public:
  explicit Setup(QWidget *parent = 0);

private:
  QWidget *low_voltage();
  QWidget *getting_started();
  QWidget *network_setup();
  QWidget *software_selection();
  QWidget *downloading();
  QWidget *download_failed();
  int download_file_xferinfo(curl_off_t dltotal, curl_off_t dlno, curl_off_t ultotal, curl_off_t ulnow);

  QWidget *failed_widget;
  QWidget *downloading_widget;
  QProgressBar *progress_bar;

signals:
  void finished(bool success);

public slots:
  void nextPage();
  void prevPage();
  void download(QString url);
};
