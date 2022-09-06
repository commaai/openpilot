#pragma once

#include <QStackedWidget>
#include <QString>
#include <QTranslator>
#include <QWidget>

class Setup : public QStackedWidget {
  Q_OBJECT

public:
  explicit Setup(QWidget *parent = 0);
  void showEvent(QShowEvent *event);

private:
  QWidget *low_voltage();
  QWidget *getting_started();
  QWidget *network_setup();
  QWidget *software_selection();
  QWidget *downloading();
  QWidget *download_failed();

  QWidget *failed_widget;
  QWidget *downloading_widget;
  QTranslator translator;

signals:
  void finished(bool success);

public slots:
  void nextPage();
  void prevPage();
  void download(QString url);
};
