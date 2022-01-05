#pragma once

#include <QStackedWidget>
#include <QString>
#include <QWidget>

#include "selfdrive/ui/ui.h"

class Setup : public QStackedWidget, public Wakeable {
  Q_OBJECT
  Q_INTERFACES(Wakeable)

public:
  explicit Setup(QWidget *parent = 0);

private:
  QWidget *low_voltage();
  QWidget *getting_started();
  QWidget *network_setup();
  QWidget *software_selection();
  QWidget *downloading();
  QWidget *download_failed();

  QWidget *failed_widget;
  QWidget *downloading_widget;

signals:
  void finished(bool success);
  void displayPowerChanged(bool on);
  void interactiveTimeout();

public slots:
  void nextPage();
  void prevPage();
  void download(QString url);
  virtual void update(const UIState &s);
};
