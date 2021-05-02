#pragma once
#include <map>

#include <QFrame>
#include <QPushButton>
#include <QLabel>
#include <QVBoxLayout>

#include "common/params.h"
#include "widgets/scrollview.h"

class Alert {
public:
  void refresh();
  bool hasAlerts() { return updateAvailable || alertCount > 0;}
  struct AlertMessage {
    std::string key;
    int severity;
    QLabel *label;
  };
  std::vector<AlertMessage> alerts;
  int alertCount = 0;
  bool updateAvailable = false;
  Params params;
};

class OffroadAlert : public QFrame {
  Q_OBJECT

public:
  explicit OffroadAlert(const Alert &alert, QWidget *parent = 0);
  void update(const Alert &alert);

signals:
  void closeAlerts();

private:
  QLabel releaseNotes;
  QPushButton rebootBtn;
  ScrollView *releaseNotesScroll;
  ScrollView *alertsScroll;
  QVBoxLayout *alerts_layout;
};
