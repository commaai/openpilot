#pragma once

#include <QScroller>
#include <QScrollArea>

class GoodScrollArea : public QScrollArea {
  Q_OBJECT

public:
  explicit GoodScrollArea(QWidget *area = nullptr);
};
