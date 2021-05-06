#pragma once

#include <QWidget>

class LayoutWidget : public QWidget {
  Q_OBJECT

public:
  explicit LayoutWidget(QLayout *l, QWidget *parent = 0);
};
