#pragma once

#include <QScrollArea>
#include <QScroller>

class ScrollView : public QScrollArea {
  Q_OBJECT

public:
  explicit ScrollView(QWidget *w = nullptr, QWidget *parent = nullptr);
protected:
  void hideEvent(QHideEvent *e);
};
