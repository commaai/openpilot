#include "selfdrive/ui/qt/widgets/layout.h"

LayoutWidget::LayoutWidget(QLayout *l, QWidget *parent) : QWidget(parent) {
  setLayout(l);
}
