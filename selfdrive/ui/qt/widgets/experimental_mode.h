#pragma once

#include <QLabel>

class ExperimentalMode : public QFrame {
  Q_OBJECT

public:
  explicit ExperimentalMode(QWidget* parent = 0);

private:
  void showEvent(QShowEvent *event) override;
};
