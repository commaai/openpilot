#pragma once

#include <QFrame>
#include <QHBoxLayout>
#include <QLabel>
#include <QPainter>
#include <QPushButton>
#include <QMouseEvent>
#include <QDebug>
#include <QStyle>

#include "common/params.h"

class ExperimentalMode : public QFrame {
  Q_OBJECT
  Q_PROPERTY(bool experimental_mode MEMBER experimental_mode);

public:
  explicit ExperimentalMode(QWidget* parent = 0);

signals:
  void openSettings();
  void pressed();
  void released();
  void clicked();

private:
  void updateStyle() {style()->unpolish(this); style()->polish(this);}

  Params params;
  bool experimental_mode;
  void showEvent(QShowEvent *event) override;
  QPushButton *button;

protected:
  void mousePressEvent(QMouseEvent *event) override {
    if (rect().contains(event->pos())) {
      emit pressed();
      qDebug() << "pressed";
      updateStyle();
    }
  }

  void mouseReleaseEvent(QMouseEvent *event) override {
    emit released();
    qDebug() << "released";
    if (rect().contains(event->pos())) {
      emit clicked();
      emit openSettings();
      qDebug() << "clicked";
    }
    updateStyle();
  }
};
