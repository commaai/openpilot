#pragma once

#include <QFrame>
#include <QStackedLayout>
#include <QWidget>

#include "selfdrive/ui/ui.h"

class WiFiPromptWidget : public QFrame {
  Q_OBJECT

public:
  explicit WiFiPromptWidget(QWidget* parent = 0);

signals:
  void openSettings(int index = 0, const QString &param = "");

public slots:
  void updateState(const UIState &s);

protected:
  QStackedLayout *stack;
};
