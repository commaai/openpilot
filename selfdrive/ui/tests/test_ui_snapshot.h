#pragma once

#include <functional>
#include <QWidget>

#include "selfdrive/ui/ui.h"

struct TestCase {
  std::function<void()> setupFunc;
  std::string name;
};

void saveWidgetAsImage(QWidget *widget, const QString &fileName);
