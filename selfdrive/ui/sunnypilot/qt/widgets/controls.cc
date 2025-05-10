/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#include "selfdrive/ui/sunnypilot/qt/widgets/controls.h"

#include <QPainter>
#include <QStyleOption>

QFrame *horizontal_line(QWidget *parent) {
  QFrame *line = new QFrame(parent);
  line->setFrameShape(QFrame::StyledPanel);
  line->setStyleSheet(R"(
    border-width: 2px;
    border-bottom-style: solid;
    border-color: gray;
  )");
  line->setFixedHeight(10);
  return line;
}

QFrame *vertical_space(int height, QWidget *parent) {
  QFrame *v_space = new QFrame(parent);
  v_space->setFrameShape(QFrame::StyledPanel);
  v_space->setFixedHeight(height);
  return v_space;
}

// AbstractControlSP

AbstractControlSP::AbstractControlSP(const QString &title, const QString &desc, const QString &icon, QWidget *parent)
    : AbstractControl(title, desc, icon, parent) {

  main_layout = new QVBoxLayout(this);
  main_layout->setMargin(0);

  hlayout = new QHBoxLayout;
  hlayout->setMargin(0);
  hlayout->setSpacing(20);

  // title
  title_label = new QPushButton(title);
  title_label->setFixedHeight(120);
  title_label->setStyleSheet("font-size: 50px; font-weight: 450; text-align: left; border: none;");
  hlayout->addWidget(title_label, 1);

  // value next to control button
  value = new ElidedLabelSP();
  value->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
  value->setStyleSheet("color: #aaaaaa");
  hlayout->addWidget(value);

  main_layout->addLayout(hlayout);

  // description
  description = new QLabel(desc);
  description->setContentsMargins(40, 20, 40, 20);
  description->setStyleSheet("font-size: 40px; color: grey");
  description->setWordWrap(true);
  description->setVisible(false);
  main_layout->addWidget(description);

  connect(title_label, &QPushButton::clicked, [=]() {
    if (!description->isVisible()) {
      emit showDescriptionEvent();
    }

    if (!description->text().isEmpty()) {
      description->setVisible(!description->isVisible());
    }
  });

  main_layout->addStretch();
}

void AbstractControlSP::hideEvent(QHideEvent *e) {
  if (description != nullptr) {
    description->hide();
  }
}

AbstractControlSP_SELECTOR::AbstractControlSP_SELECTOR(const QString &title, const QString &desc, const QString &icon, QWidget *parent)
    : AbstractControlSP(title, desc, icon, parent) {

  if (title_label != nullptr) {
    delete title_label;
    title_label = nullptr;
  }

  if (description != nullptr) {
    delete description;
    description = nullptr;
  }

  if (value != nullptr) {
    ReplaceWidget(value, new QWidget());
    value = nullptr;
  }

  QLayoutItem *item;
  while ((item = main_layout->takeAt(0)) != nullptr) {
    if (item->widget()) {
      delete item->widget();
    }
    delete item;
  }

  main_layout->setMargin(0);

  hlayout = new QHBoxLayout;
  hlayout->setMargin(0);
  hlayout->setSpacing(0);

  // title
  if (!title.isEmpty()) {
    title_label = new QPushButton(title);
    title_label->setFixedHeight(120);
    title_label->setStyleSheet("font-size: 50px; font-weight: 450; text-align: left; border: none; padding: 0 0 0 0");
    main_layout->addWidget(title_label, 1);

    connect(title_label, &QPushButton::clicked, [=]() {
      if (!description->isVisible()) {
        emit showDescriptionEvent();
      }

      if (!description->text().isEmpty()) {
        bool isVisible = !description->isVisible();
        description->setVisible(isVisible);

        if (isVisible && spacingItem) {
          main_layout->removeItem(spacingItem);
        } else if (!isVisible && spacingItem != nullptr && main_layout->indexOf(spacingItem) == -1) {
          main_layout->insertItem(main_layout->indexOf(description), spacingItem);
        }
      }
    });
  } else {
    main_layout->addSpacing(20);
  }

  main_layout->addLayout(hlayout);
  if (!desc.isEmpty() && spacingItem != nullptr && main_layout->indexOf(spacingItem) == -1) {
    main_layout->insertItem(main_layout->count(), spacingItem);
  }

  // description
  description = new QLabel(desc);
  description->setContentsMargins(40, 20, 40, 20);
  description->setStyleSheet("font-size: 40px; color: grey");
  description->setWordWrap(true);
  description->setVisible(false);
  main_layout->addWidget(description);

  main_layout->addStretch();
}

void AbstractControlSP_SELECTOR::hideEvent(QHideEvent *e) {
  if (description != nullptr) {
    description->hide();
  }

  if (spacingItem != nullptr && main_layout->indexOf(spacingItem) == -1) {
    main_layout->insertItem(main_layout->indexOf(description), spacingItem);
  }
}

// controls

ButtonControlSP::ButtonControlSP(const QString &title, const QString &text, const QString &desc, QWidget *parent)
    : AbstractControlSP(title, desc, "", parent) {

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
    QPushButton:pressed {
      background-color: #4a4a4a;
    }
    QPushButton:disabled {
      color: #33E4E4E4;
    }
  )");
  btn.setFixedSize(250, 100);
  QObject::connect(&btn, &QPushButton::clicked, this, &ButtonControlSP::clicked);
  hlayout->addWidget(&btn);
}

// ElidedLabelSP

ElidedLabelSP::ElidedLabelSP(QWidget *parent) : ElidedLabelSP({}, parent) {
}

ElidedLabelSP::ElidedLabelSP(const QString &text, QWidget *parent) : QLabel(text.trimmed(), parent) {
  setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
  setMinimumWidth(1);
}

void ElidedLabelSP::resizeEvent(QResizeEvent *event) {
  QLabel::resizeEvent(event);
  lastText_ = elidedText_ = "";
}

void ElidedLabelSP::paintEvent(QPaintEvent *event) {
  const QString curText = text();
  if (curText != lastText_) {
    elidedText_ = fontMetrics().elidedText(curText, Qt::ElideRight, contentsRect().width());
    lastText_ = curText;
  }

  QPainter painter(this);
  drawFrame(&painter);
  QStyleOption opt;
  opt.initFrom(this);
  style()->drawItemText(&painter, contentsRect(), alignment(), opt.palette, isEnabled(), elidedText_, foregroundRole());
}

// ParamControlSP

ParamControlSP::ParamControlSP(const QString &param, const QString &title, const QString &desc, const QString &icon, QWidget *parent)
    : ToggleControlSP(title, desc, icon, false, parent) {

  key = param.toStdString();
  QObject::connect(this, &ParamControlSP::toggleFlipped, this, &ParamControlSP::toggleClicked);

  hlayout->removeWidget(&toggle);
  hlayout->insertWidget(0, &toggle);

  hlayout->removeWidget(this->icon_label);
  hlayout->insertWidget(1, this->icon_label);
}

void ParamControlSP::toggleClicked(bool state) {
  auto do_confirm = [this]() {
    QString content("<body><h2 style=\"text-align: center;\">" + title_label->text() + "</h2><br>"
                    "<p style=\"text-align: center; margin: 0 128px; font-size: 50px;\">" + getDescription() + "</p></body>");
    return ConfirmationDialog(content, tr("Enable"), tr("Cancel"), true, this).exec();
  };

  bool confirmed = store_confirm && params.getBool(key + "Confirmed");
  if (!confirm || confirmed || !state || do_confirm()) {
    if (store_confirm && state) params.putBool(key + "Confirmed", true);
    params.putBool(key, state);
    setIcon(state);
  } else {
    toggle.togglePosition();
  }
}
