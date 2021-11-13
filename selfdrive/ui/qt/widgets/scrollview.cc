#include "selfdrive/ui/qt/widgets/scrollview.h"

#include <QScrollBar>
#include <QScroller>

ScrollView::ScrollView(QWidget *w, QWidget *parent) : QScrollArea(parent) {
  setWidget(w);
  setWidgetResizable(true);
  setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  setStyleSheet("background-color: transparent;");

  QString style = R"(
    QScrollBar:vertical {
      border: none;
      background: transparent;
      width: 10px;
      margin: 0;
    }
    QScrollBar::handle:vertical {
      min-height: 0px;
      border-radius: 5px;
      background-color: white;
    }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
      height: 0px;
    }
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
      background: none;
    }
  )";
  verticalScrollBar()->setStyleSheet(style);
  horizontalScrollBar()->setStyleSheet(style);

  QScroller *scroller = QScroller::scroller(this->viewport());
  QScrollerProperties sp = scroller->scrollerProperties();

  sp.setScrollMetric(QScrollerProperties::VerticalOvershootPolicy, QVariant::fromValue<QScrollerProperties::OvershootPolicy>(QScrollerProperties::OvershootAlwaysOff));
  sp.setScrollMetric(QScrollerProperties::HorizontalOvershootPolicy, QVariant::fromValue<QScrollerProperties::OvershootPolicy>(QScrollerProperties::OvershootAlwaysOff));

  scroller->grabGesture(this->viewport(), QScroller::LeftMouseButtonGesture);
  scroller->setScrollerProperties(sp);
}

void ScrollView::hideEvent(QHideEvent *e) {
  verticalScrollBar()->setValue(0);
}
