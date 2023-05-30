#pragma once

#include <QFrame>
#include <QStackedLayout>
#include <QWidget>

#include "selfdrive/ui/ui.h"

class WiFiPromptWidget : public QFrame {
  Q_OBJECT

public:
  explicit WiFiPromptWidget(QWidget* parent = 0);

public slots:
  void updateState(const UIState &s);

protected:
  QStackedLayout *stack;
};
