#pragma once

#include <QFrame>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>

#include "selfdrive/common/params.h"
#include "selfdrive/ui/qt/widgets/toggle.h"

QFrame *horizontal_line(QWidget *parent = nullptr);
class AbstractControl : public QFrame {
  Q_OBJECT

public:
  void setDescription(const QString &desc) {
    if(description) description->setText(desc);
  }

signals:
  void showDescription();

protected:
  AbstractControl(const QString &title, const QString &desc = "", const QString &icon = "", QWidget *parent = nullptr);
  void hideEvent(QHideEvent *e) override;

  QSize minimumSizeHint() const override {
    QSize size = QFrame::minimumSizeHint();
    size.setHeight(120);
    return size;
  };

  QHBoxLayout *hlayout;
  QPushButton *title_label;
  QLabel *description = nullptr;
};

// widget to display a value
class LabelControl : public AbstractControl {
  Q_OBJECT

public:
  LabelControl(const QString &title, const QString &text = "", const QString &desc = "", QWidget *parent = nullptr) : AbstractControl(title, desc, "", parent) {
    label.setText(text);
    label.setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    hlayout->addWidget(&label);
  }
  void setText(const QString &text) { label.setText(text); }

private:
  QLabel label;
};

// widget for a button with a label
class ButtonControl : public AbstractControl {
  Q_OBJECT

public:
  template <typename Functor>
  ButtonControl(const QString &title, const QString &text, const QString &desc, Functor functor, const QString &icon = "", QWidget *parent = nullptr) : AbstractControl(title, desc, icon, parent) {
    btn.setText(text);
    btn.setStyleSheet(R"(
      QPushButton {
        padding: 0;
        border-radius: 50px;
        font-size: 35px;
        font-weight: 500;
        color: #E4E4E4;
        background-color: #393939;
      }
      QPushButton:disabled {
        color: #33E4E4E4;
      }
    )");
    btn.setFixedSize(250, 100);
    QObject::connect(&btn, &QPushButton::released, functor);
    hlayout->addWidget(&btn);
  }
  void setText(const QString &text) { btn.setText(text); }

public slots:
  void setEnabled(bool enabled) {
    btn.setEnabled(enabled);
  };

private:
  QPushButton btn;
};

class ToggleControl : public AbstractControl {
  Q_OBJECT

public:
  ToggleControl(const QString &title, const QString &desc = "", const QString &icon = "", const bool state = false, QWidget *parent = nullptr) : AbstractControl(title, desc, icon, parent) {
    toggle.setFixedSize(150, 100);
    if (state) {
      toggle.togglePosition();
    }
    hlayout->addWidget(&toggle);
    QObject::connect(&toggle, &Toggle::stateChanged, this, &ToggleControl::toggleFlipped);
  }

  void setEnabled(bool enabled) { toggle.setEnabled(enabled); }

signals:
  void toggleFlipped(bool state);

protected:
  Toggle toggle;
};

// widget to toggle params
class ParamControl : public ToggleControl {
  Q_OBJECT

public:
  ParamControl(const QString &param, const QString &title, const QString &desc, const QString &icon, QWidget *parent = nullptr) : ToggleControl(title, desc, icon, false, parent) {
    if (params.getBool(param.toStdString().c_str())) {
      toggle.togglePosition();
    }
    QObject::connect(this, &ToggleControl::toggleFlipped, [=](bool state) {
      params.putBool(param.toStdString().c_str(), state);
    });
  }

private:
  Params params;
};
