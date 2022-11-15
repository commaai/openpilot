#pragma once

#include <QLabel>
#include <QPushButton>

#include "common/params.h"

class ExperimentalMode : public QPushButton {
  Q_OBJECT
  Q_PROPERTY(bool experimental_mode MEMBER experimental_mode);

public:
  explicit ExperimentalMode(QWidget* parent = 0);

signals:
  void openSettings(int index = 0);

private:
  void showEvent(QShowEvent *event) override;

  Params params;
  bool experimental_mode;
  QPixmap experimental_pixmap;
  QPixmap chill_pixmap;
  QLabel *mode_label;
  QLabel *mode_icon;
};
