#pragma once

#include <QFrame>
#include <QLabel>

class ClickableLabel : public QFrame {
  Q_OBJECT

 public:
  ClickableLabel(const QString &title, const QString &desc, QWidget *control,
                 const QString &icon = "", bool bottom_line = true);

  void mousePressEvent(QMouseEvent *event) override {
    pressed = true;
  }
  void mouseMoveEvent(QMouseEvent *event) override {
    if (pressed) dragging = true;
  }
  void mouseReleaseEvent(QMouseEvent *event) override {
    if (!dragging && desc_label) {
      desc_label->setVisible(!desc_label->isVisible());
    }
    pressed = dragging = false;
  }

 private:
  bool pressed = false, dragging = false;
  QLabel *desc_label = nullptr;
};
