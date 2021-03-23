#pragma once

#include <QFrame>
#include <QStackedWidget>
#include <QPushButton>
#include <QStringList>
#include <vector>

struct Alert {
  QString text;
  int severity;
};

inline bool operator==(const Alert &l, const Alert &r) {
  return l.text == r.text && l.severity == r.severity;
}

class OffroadAlert : public QFrame {
  Q_OBJECT

public:
  explicit OffroadAlert(QWidget *parent = 0);
  std::vector<Alert> alerts;
  QStringList alert_keys;
  bool updateAvailable = false;

private:
  QStackedWidget *alerts_stack;
  QPushButton *reboot_btn;
  std::vector<Alert> parse_alerts();

signals:
  void closeAlerts();

public slots:
  void refresh();
};
