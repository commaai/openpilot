#include "scrollview.hpp"
#include <QScrollBar>

ScrollView::ScrollView(QWidget *w, QWidget *parent) : QScrollArea(parent){
  setWidget(w);
  setWidgetResizable(true);
  setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  setStyleSheet("ScrollView { background-color:transparent; }");

  QString style = R"(
  QScrollBar:vertical {
    border: 0px solid #999999;
    background:transparent;
    width:10px;
    margin: 0px 0px 0px 0px;
  }
  QScrollBar::handle:vertical {
    min-height: 0px;
    border: 0px solid red;
    border-radius: 4px;
    background-color: white;
  }
  QScrollBar::add-line:vertical {
    height: 0px;
    subcontrol-position: bottom;
    subcontrol-origin: margin;
  }
  QScrollBar::sub-line:vertical {
    height: 0 px;
    subcontrol-position: top;
    subcontrol-origin: margin;
  })";

  verticalScrollBar()->setStyleSheet(style);
  horizontalScrollBar()->setStyleSheet(style);

  QScroller *scroller = QScroller::scroller(this->viewport());
  QScrollerProperties sp = scroller->scrollerProperties();

  sp.setScrollMetric(QScrollerProperties::VerticalOvershootPolicy, QVariant::fromValue<QScrollerProperties::OvershootPolicy>(QScrollerProperties::OvershootAlwaysOff));
  sp.setScrollMetric(QScrollerProperties::HorizontalOvershootPolicy, QVariant::fromValue<QScrollerProperties::OvershootPolicy>(QScrollerProperties::OvershootAlwaysOff));

  scroller->grabGesture(this->viewport(), QScroller::LeftMouseButtonGesture);
  scroller->setScrollerProperties(sp);
}
