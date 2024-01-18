#pragma once

#include "selfdrive/ui/ui.h"
#include <QLabel>
#include <QPropertyAnimation>

class AssistantOverlay : public QLabel {
  Q_OBJECT

public:
  explicit AssistantOverlay(QWidget *parent = nullptr);
  
  void updateText(const QString &newText);
  void animateShow();
  void animateHide();

private:
  QTimer *hideTimer;
  QPropertyAnimation *showAnimation;
  QPropertyAnimation *hideAnimation;
  int parentCenterX;
  int finalWidth;
  int startX;
  
private slots:
  void updateState(const UIState &s);

};
