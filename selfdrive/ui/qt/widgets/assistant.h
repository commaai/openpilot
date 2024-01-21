#pragma once

#include "selfdrive/ui/ui.h"
#include <QLabel>
#include <QPropertyAnimation>

class AssistantOverlay : public QLabel {
  Q_OBJECT

public:
  explicit AssistantOverlay(QWidget *parent = nullptr);
  void animateOverlay(bool show);

private:
  void updateText(const QString text);
  void startHideTimer();
  QTimer *hideTimer;
  QPropertyAnimation *showAnimation;
  QPropertyAnimation *hideAnimation;
  int finalWidth;

private slots:
  void updateState(const UIState &s);

};
