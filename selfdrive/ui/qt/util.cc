#include "selfdrive/ui/qt/util.h"
#include <QStyleOption>
#include <QDebug>

ClickableWidget::ClickableWidget(QWidget *parent) : QWidget(parent) { }

void ClickableWidget::mouseReleaseEvent(QMouseEvent *event) {
  emit clicked();
}


// Fix stylesheets
void ClickableWidget::paintEvent(QPaintEvent *)  {
    QStyleOption opt;
    opt.init(this);
    QPainter p(this);
    style()->drawPrimitive(QStyle::PE_Widget, &opt, &p, this);
}
