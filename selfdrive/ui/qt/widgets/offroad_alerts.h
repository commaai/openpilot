#pragma once

#include <map>

#include <QFrame>
#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>

#include "selfdrive/ui/qt/widgets/scrollview.h"

class OffroadAlert : public QFrame {
  Q_OBJECT

public:
  explicit OffroadAlert(QWidget *parent = 0);
  int alertCount = 0;
  bool updateAvailable;

private:
  void updateAlerts();

  std::map<QString, QLabel*> alerts;

  QLabel releaseNotes;
  QPushButton rebootBtn;
  ScrollView *alertsScroll;
  ScrollView *releaseNotesScroll;
  QVBoxLayout *alerts_layout;

signals:
  void closeAlerts();

public slots:
  void refresh();
};
