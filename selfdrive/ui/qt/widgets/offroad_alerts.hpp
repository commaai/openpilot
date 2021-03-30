#pragma once
#include <map>

#include <QFrame>
#include <QPushButton>
#include <QLabel>

#include "common/params.h"

class OffroadAlert : public QFrame {
  Q_OBJECT

public:
  explicit OffroadAlert(QWidget *parent = 0);
  int alertCount = 0;
  bool updateAvailable;

private:
  Params params;
  QLabel releaseNotes;
  std::map<std::string, QLabel*> alerts;
  QPushButton rebootBtn;
  void updateAlerts();

signals:
  void closeAlerts();

public slots:
  void refresh();
};
