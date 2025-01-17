#pragma once

#include <QLabel>

class TipLabel : public QLabel {
  Q_OBJECT

public:
  TipLabel(QWidget *parent = nullptr);
  void showText(const QPoint &pt, const QString &sec, QWidget *w, const QRect &rect);
  void paintEvent(QPaintEvent *ev) override;
};
