#pragma once

#include <curl/curl.h>

#include <QDebug>
#include <QLabel>
#include <QProgressBar>
#include <QPushButton>
#include <QStackedWidget>
#include <QThread>

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
  bool download_file(const QString &url, const QString &out_fn);
  QString download(const QString &url, const QString &hash, const QString &name);
  int download_file_xferinfo(curl_off_t dltotal, curl_off_t dlno, curl_off_t ultotal, curl_off_t ulnow);

  CURL *curl = nullptr;
  QString recovery_hash;
  QString recovery_fn, ota_fn;
  size_t recovery_len;
};

class UpdaterWidnow : public QStackedWidget {
  Q_OBJECT
public:
  UpdaterWidnow(QWidget *parent = nullptr);
  virtual ~UpdaterWidnow();
private:
  QWidget *confirmationPage();
  QWidget *progressPage();
  QWidget *batteryPage();
  QWidget *errPage();

  UpdaterThread thread;
  QLabel *progressTitle;
  QLabel *errLabel;
  QLabel *batteryContext;
  QProgressBar *progress_bar;
};
