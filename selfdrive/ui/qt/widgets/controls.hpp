#pragma once

#include <QFrame>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>

#include "common/params.h"
#include "toggle.hpp"

QFrame *horizontal_line(QWidget *parent = nullptr);

class AbstractControl : public QFrame {
  Q_OBJECT

protected:
  AbstractControl(const QString &title, const QString &desc = "", const QString &icon = "");

  void setControlWidget(QWidget *w) {
    assert(control_widget == nullptr);
    hboxLayout->addWidget(w);
    control_widget = w;
  }

  QSize minimumSizeHint() const override {
    QSize size = QFrame::minimumSizeHint();
    size.setHeight(120);
    return size;
  };

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

private:
  bool pressed = false, dragging = false;
  QHBoxLayout *hboxLayout = nullptr;
  QLabel *title_label = nullptr, *desc_label = nullptr;
  QWidget *control_widget = nullptr;
};

class LabelControl : public AbstractControl {
  Q_OBJECT

public:
  LabelControl(const QString &title, const QString &text, const QString &desc = "") : AbstractControl(title, desc, "") {
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

  void setEnabled(bool enabled) { toggle.setEnabled(enabled); }

 private:
  Toggle toggle;
};
