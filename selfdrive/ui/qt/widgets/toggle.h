#pragma once

#include <QAbstractButton>
#include <QMouseEvent>
#include <QPropertyAnimation>

class Toggle : public QAbstractButton {
  Q_OBJECT
  Q_PROPERTY(int offset_circle READ offset_circle WRITE set_offset_circle CONSTANT)

public:
  Toggle(QWidget* parent = nullptr);
  void togglePosition();
  bool on;
  int animation_duration = 150;
  int immediateOffset = 0;
  int offset_circle() const {
    return _x_circle;
  }

  void set_offset_circle(int o) {
    _x_circle = o;
    update();
  }
  bool getEnabled();
  virtual void setEnabled(bool value);

protected:
  void paintEvent(QPaintEvent*) override;
  void mouseReleaseEvent(QMouseEvent*) override;
  void enterEvent(QEvent*) override;

  QColor circleColor;
  QColor green;
  bool enabled = true;
  int _x_circle, _y_circle;
  int _height, _radius;
  int _height_rect, _y_rect;
  QPropertyAnimation *_anim = nullptr;

signals:
  void stateChanged(bool new_state);
};
