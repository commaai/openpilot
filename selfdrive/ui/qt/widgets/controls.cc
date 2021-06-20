#include "selfdrive/ui/qt/widgets/controls.h"

#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>

#include "selfdrive/common/params.h"
#include "selfdrive/ui/qt/widgets/toggle.h"

QFrame *horizontal_line(QWidget *parent) {
  QFrame *line = new QFrame(parent);
  line->setFrameShape(QFrame::StyledPanel);
  line->setStyleSheet(R"(
    margin-left: 40px;
    margin-right: 40px;
    border-width: 1px;
    border-bottom-style: solid;
    border-color: gray;
  )");
  line->setFixedHeight(2);
  return line;
}

AbstractControl::AbstractControl(const QString &title, const QString &desc,
                                 QWidget *parent) : QFrame(parent) {
  main_layout = new QVBoxLayout(this);
  main_layout->setMargin(0);

  hlayout = new QHBoxLayout;
  hlayout->setMargin(0);
  hlayout->setSpacing(20);

  // title
  title_label = new QPushButton(title);
  title_label->setStyleSheet("font-size: 50px; font-weight: 400; text-align: left;");
  hlayout->addWidget(title_label, 1);

  controls_layout = new QHBoxLayout();
  hlayout->addLayout(controls_layout, 0);

  main_layout->addLayout(hlayout);

  // description
  if (!desc.isEmpty()) {
    setDescription(desc);
  }

  setStyleSheet("background-color: transparent;");
}

QSize AbstractControl::minimumSizeHint() const {
  QSize size = QFrame::minimumSizeHint();
  size.setHeight(120);
  return size;
};

void AbstractControl::setIcon(const QString &icon) {
  if (!icon_label) {
    icon_label = new QLabel();
    icon_label->setSizePolicy(QSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed));
    hlayout->insertWidget(0, icon_label);
  }
  icon_label->setPixmap(QPixmap(icon).scaledToWidth(80, Qt::SmoothTransformation));
}

void AbstractControl::setDescription(const QString &text) {
  if (desc_label == nullptr) {
    desc_label = new QLabel();
    desc_label->setContentsMargins(40, 20, 40, 20);
    desc_label->setStyleSheet("font-size: 40px; color:grey");
    desc_label->setWordWrap(true);
    desc_label->setVisible(false);
    main_layout->addWidget(desc_label);

    connect(title_label, &QPushButton::clicked, [=]() {
      if (!desc_label->isVisible()) {
        emit showDescription();
      }
      desc_label->setVisible(!desc_label->isVisible());
    });
  }
  desc_label->setText(text);
};

QString AbstractControl::description() const {
  return desc_label ? desc_label->text() : "";
}

void AbstractControl::hideEvent(QHideEvent *e) {
  if (desc_label != nullptr) {
    desc_label->hide();
  }
}

// ButtonControl

ButtonControl::ButtonControl(const QString &title, const QString &text,
                             const QString &desc, QWidget *parent) : AbstractControl(title, desc, parent) {
  btn = new QPushButton;
  btn->setText(text);
  btn->setStyleSheet(R"(
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
  btn->setFixedSize(250, 100);
  QObject::connect(btn, &QPushButton::released, this, &ButtonControl::released);
  controls_layout->addWidget(btn);
}

void ButtonControl::setText(const QString &text) { btn->setText(text); }
QString ButtonControl::text() const { return btn->text(); }
void ButtonControl::setEnabled(bool enabled) { btn->setEnabled(enabled); };

// LabelControl

LabelControl::LabelControl(const QString &title, const QString &text, const QString &desc,
                           QWidget *parent) : AbstractControl(title, desc, parent) {
  label = new QLabel;
  label->setText(text);
  label->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
  controls_layout->addWidget(label);
}

void LabelControl::setText(const QString &text) { label->setText(text); }

// ToggleControl

ToggleControl::ToggleControl(const QString &title, const QString &desc, const QString &icon,
                             const bool state, QWidget *parent) : AbstractControl(title, desc, parent) {
  setIcon(icon);
  toggle = new Toggle;
  toggle->setFixedSize(150, 100);
  if (state) {
    toggle->togglePosition();
  }
  controls_layout->addWidget(toggle);
  QObject::connect(toggle, &Toggle::stateChanged, this, &ToggleControl::toggleFlipped);
}

void ToggleControl::setEnabled(bool enabled) { toggle->setEnabled(enabled); }

// ParamControl

ParamControl::ParamControl(const QString &param, const QString &title, const QString &desc,
                           const QString &icon, QWidget *parent) : ToggleControl(title, desc, icon, false, parent) {
  if (Params().getBool(param.toStdString().c_str())) {
    toggle->togglePosition();
  }
  QObject::connect(this, &ToggleControl::toggleFlipped, [=](bool state) {
    Params().putBool(param.toStdString().c_str(), state);
  });
}
