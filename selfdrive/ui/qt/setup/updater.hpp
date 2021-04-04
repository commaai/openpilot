#pragma once

#include <curl/curl.h>

#include <QDebug>
#include <QLabel>
#include <QProgressBar>
#include <QPushButton>
#include <QStackedWidget>
#include <QThread>
#include <string>

class UpdaterThread : public QThread {
  Q_OBJECT
public:
  UpdaterThread(QObject *parent);

signals:
  void progressText(const QString &);
  void progressPos(int);
  void error(const QString &);
  void lowBattery(int);

private:
  void run() override;
  void checkBattery();
  bool download_stage();
  bool download_file(const std::string &url, const std::string &out_fn);
  std::string download(const std::string &url, const std::string &hash, const std::string &name);
  int download_file_xferinfo(curl_off_t dltotal, curl_off_t dlno, curl_off_t ultotal, curl_off_t ulnow);

  CURL *curl = nullptr;
  std::string recovery_hash;
  std::string recovery_fn, ota_fn;
  size_t recovery_len;
};

class UpdaterWidnow : public QStackedWidget {
  Q_OBJECT
public:
  UpdaterWidnow(QWidget *parent = nullptr);

private:
  QWidget *confirmationPage();
  QWidget *progressPage();
  QWidget *errPage();
  QWidget *batteryPage();

  UpdaterThread thread;
  QLabel *progressTitle;
  QLabel *errLabel;
  QLabel *batteryContext;
  QProgressBar *progress_bar;
};
