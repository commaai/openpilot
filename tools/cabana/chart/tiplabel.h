#pragma once

#include <QLabel>

class TipLabel : public QLabel {
public:
  TipLabel(QWidget *parent = nullptr);
  void showText(const QPoint &pt, const QString &sec, int right_edge);
  void paintEvent(QPaintEvent *ev) override;
};
