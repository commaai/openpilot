#pragma once

#include <QLabel>
#include <QStackedWidget>
#include <QString>
#include <QWidget>

enum DownloadResult {
  ok,
  notExecutable,
  error,
};

class Setup : public QStackedWidget {
  Q_OBJECT

public:
  explicit Setup(QWidget *parent = 0);

private:
  QWidget *low_voltage();
  QWidget *getting_started();
  QWidget *network_setup();
  QWidget *downloading();
  QWidget *download_failed();
  QWidget *invalid_url(QLabel *url);

  QWidget *failed_widget;
  QWidget *invalid_url_widget;
  QWidget *downloading_widget;

signals:
  void complete(const DownloadResult &result, const QString &url);

public slots:
  void nextPage();
  void prevPage();
  void download(QString url);
};
