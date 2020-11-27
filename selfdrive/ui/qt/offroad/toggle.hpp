#pragma once
#include <QtWidgets>

class Toggle : public QAbstractButton {
  Q_OBJECT
  Q_PROPERTY(int offset_circle READ offset_circle WRITE set_offset_circle)

public:
  Toggle(QWidget* parent = nullptr);
  void togglePosition();

  int offset_circle() const {
    return _x_circle;
  }

  void set_offset_circle(int o) {
    _x_circle = o;
    update();
  }

protected:
  void paintEvent(QPaintEvent*) override;
  void mouseReleaseEvent(QMouseEvent*) override;
  void enterEvent(QEvent*) override;

private:
  bool _on;
  int _x_circle, _y_circle;
  int _height, _radius;
  int _height_rect, _y_rect;
  QPropertyAnimation *_anim = nullptr;

signals:
  void stateChanged(int new_state);
};
