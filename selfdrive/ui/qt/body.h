#pragma once

#include <QGridLayout>
#include <QLabel>
#include <QPainter>

#include "selfdrive/ui/ui.h"

class FaceDotMatrix : public QWidget {
  Q_OBJECT

public:
  FaceDotMatrix(QWidget* parent = 0);

protected:
  virtual void paintEvent(QPaintEvent *event);
};


class Eye : public FaceDotMatrix {
  Q_OBJECT

public:
  Eye(QWidget* parent = 0);

protected:
  void paintEvent(QPaintEvent *event);
};

class Mouth : public FaceDotMatrix {
  Q_OBJECT

public:
  Mouth(QWidget* parent = 0);

protected:
  void paintEvent(QPaintEvent *event);
};

class BodyWindow : public QWidget {
  Q_OBJECT

public:
  BodyWindow(QWidget* parent = 0);

private:
  QGridLayout *layout;
  Eye *leftEye;
  Eye *rightEye;
  Mouth *mouth;

private slots:
  void updateState(const UIState &s);
};
