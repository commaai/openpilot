#pragma once

#include <QScrollArea>

class ScrollView : public QScrollArea {
  Q_OBJECT

public:
  explicit ScrollView(QWidget *w = nullptr, QWidget *parent = nullptr);
protected:
  void hideEvent(QHideEvent *e) override;
};
