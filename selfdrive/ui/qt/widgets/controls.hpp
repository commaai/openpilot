#pragma once

#include <QFrame>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>

#include "common/params.h"
#include "toggle.hpp"

class AbstractControl : public QFrame {
  Q_OBJECT
 public:
  void setEnabled(bool enabled) {
    title_label->setEnabled(enabled);
    if (control) {
      control->setEnabled(enabled);
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
  void setControl(QWidget *w) {
    assert(control == nullptr);
    hboxLayout->addWidget(w);
    control = w;
  }

 private:
  QHBoxLayout *hboxLayout;
  bool pressed = false, dragging = false;
  QLabel *title_label = nullptr, *desc_label = nullptr;
  QWidget *control = nullptr;
};

class LabelControl : public AbstractControl {
  Q_OBJECT
 public:
  LabelControl(const QString &title, const QString &text) : AbstractControl(title) {
    label = new QLabel(text);
    label->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    setControl(label);
  }
  void setText(const QString &text) {
    label->setText(text);
  }

 private:
  QLabel *label;
};

class ButtonControl : public AbstractControl {
  Q_OBJECT
 public:
  template <typename Functor>
  ButtonControl(const QString &title, const QString &text, const QString &desc, Functor functor) : AbstractControl(title, desc) {
    btn = new QPushButton(text);
    btn->setStyleSheet(R"(font-size: 30px;border-radius: 30px;min-width: 220px; max-width: 220px;)");
    QObject::connect(btn, &QPushButton::released, functor);
    setControl(btn);
  }
  void setText(const QString &text) {
    btn->setText(text);
  }

 private:
  QPushButton *btn;
};

class ToggleControl : public AbstractControl {
  Q_OBJECT
 public:
  ToggleControl(const QString &param, const QString &title, const QString &desc, const QString &icon) : AbstractControl(title, desc, icon) {
    toggle = new Toggle(this);
    toggle->setFixedSize(150, 100);

    // set initial state from param
    if (Params().read_db_bool(param.toStdString().c_str())) {
      toggle->togglePosition();
    }

    QObject::connect(toggle, &Toggle::stateChanged, [=](int state) {
      char value = state ? '1' : '0';
      Params().write_db_value(param.toStdString().c_str(), &value, 1);
    });

    setControl(toggle);
  }

 private:
  Toggle *toggle;
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
