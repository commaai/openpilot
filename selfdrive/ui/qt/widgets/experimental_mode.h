#pragma once

#include <QFrame>
#include <QHBoxLayout>
#include <QLabel>
#include <QPainter>
#include <QPushButton>
#include <QMouseEvent>
#include <QDebug>
#include <QStyle>

#include "common/params.h"

class ExperimentalMode : public QPushButton {
  Q_OBJECT
  Q_PROPERTY(bool experimental_mode MEMBER experimental_mode);

public:
  explicit ExperimentalMode(QWidget* parent = 0);

signals:
  void openSettings(int index = 0);

private:
  void updateStyle() {style()->unpolish(this); style()->polish(this);}

  Params params;
  bool experimental_mode;
  void showEvent(QShowEvent *event) override;
};
