#pragma once

#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QFrame>
#include <QLabel>
#include <QPushButton>

#include "common/params.h"
#include "toggle.hpp"

class AbstractControl : public QFrame {
  Q_OBJECT
 public:
  void setEnabled(bool enabled) {
    title_label->setEnabled(enabled);
    if (control_widget) {
      control_widget->setEnabled(enabled);
    }
    if (desc_label) { 
      desc_label->setEnabled(enabled);
    }
  }

 protected:
  AbstractControl(const QString &title, const QString &desc = "", const QString &icon = "", bool bottom_line = true);
  void mousePressEvent(QMouseEvent *event) override { pressed = true; }
  void mouseMoveEvent(QMouseEvent *event) override {
    if (pressed) dragging = true;
  }
  void mouseReleaseEvent(QMouseEvent *event) override {
    if (!dragging && desc_label) {
      desc_label->setVisible(!desc_label->isVisible());
    }
    pressed = dragging = false;
  }
  void setControlWidget(QWidget *w) {
    assert(control_widget == nullptr);
    hboxLayout->addWidget(w);
    control_widget = w;
  }

 private:
  bool pressed = false, dragging = false;
  QHBoxLayout *hboxLayout = nullptr;
  QLabel *title_label = nullptr, *desc_label = nullptr;
  QWidget *control_widget = nullptr;
};

class LabelControl : public AbstractControl {
  Q_OBJECT
 public:
  LabelControl(const QString &title, const QString &text, const QString &desc = "") : AbstractControl(title, desc) {
    label.setText(text);
    label.setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    setControlWidget(&label);
  }
  void setText(const QString &text) { label.setText(text); }

 private:
  QLabel label;
};

class ButtonControl : public AbstractControl {
  Q_OBJECT
 public:
  template <typename Functor>
  ButtonControl(const QString &title, const QString &text, const QString &desc, Functor functor, const QString &icon = "") : AbstractControl(title, desc, icon) {
    btn.setText(text);
    btn.setStyleSheet(R"(padding: 0; height: 80px; border-radius: 15px;background-color: #393939;font-size: 30px;min-width: 220px; max-width: 220px;)");
    QObject::connect(&btn, &QPushButton::released, functor);
    setControlWidget(&btn);
  }
  void setText(const QString &text) { btn.setText(text); }

 private:
  QPushButton btn;
};

class ToggleControl : public AbstractControl {
  Q_OBJECT
 public:
  ToggleControl(const QString &param, const QString &title, const QString &desc, const QString &icon) : AbstractControl(title, desc, icon) {
    toggle.setFixedSize(150, 100);
    // set initial state from param
    if (Params().read_db_bool(param.toStdString().c_str())) {
      toggle.togglePosition();
    }
    QObject::connect(&toggle, &Toggle::stateChanged, [=](int state) {
      char value = state ? '1' : '0';
      Params().write_db_value(param.toStdString().c_str(), &value, 1);
    });
    setControlWidget(&toggle);
  }

 private:
  Toggle toggle;
};

class ScrollControl : public QScrollArea {
  Q_OBJECT
 public:
  ScrollControl() : QScrollArea() {
    setStyleSheet(R"(background-color: transparent;)");
    setWidgetResizable(true);
    setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    QScroller::grabGesture(this, QScroller::TouchGesture);
  }
};

QFrame *horizontal_line(QWidget *parent = nullptr);
