#pragma once

#include <QFrame>
#include <QPushButton>
#include <QStringList>
#include <QVBoxLayout>
#include <QLabel>
#include <memory>
#include <vector>
struct Alert {
  QString text;
  int severity;
};

class OffroadAlert : public QFrame {
  Q_OBJECT

public:
  explicit OffroadAlert(QWidget *parent = 0);
  QVector<Alert> alerts;
  bool updateAvailable;

private:
  QWidget *alert_widget;
  QStringList alert_keys;
  QPushButton *reboot_btn;
  std::vector<std::unique_ptr<QLabel>> labels;
  void parse_alerts();

signals:
  void closeAlerts();

public slots:
  void refresh();
};
