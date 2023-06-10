#pragma once

#include <QLabel>
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
  QWidget *downloading();
  QWidget *download_failed(QLabel *url, QLabel *body);

  QWidget *failed_widget;
  QWidget *downloading_widget;

signals:
  void finished(const QString &url, const QString &error = "");

public slots:
  void nextPage();
  void prevPage();
  void download(QString url);
};
