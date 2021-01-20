#include <QPushButton>

#include "input_field.hpp"

InputField::InputField(QWidget *parent, int minTextLength): QWidget(parent), minTextLength(minTextLength) {
  layout = new QGridLayout();
  layout->setSpacing(30);

  label = new QLabel(this);
  label->setStyleSheet(R"(font-size: 55px;)");
  layout->addWidget(label, 0, 0, Qt::AlignVCenter | Qt::AlignLeft);
  layout->setColumnStretch(0, 1);

  QPushButton* cancel = new QPushButton("Cancel");
  cancel->setFixedSize(300, 150);
  cancel->setStyleSheet(R"(padding: 0;)");
  layout->addWidget(cancel, 0, 1, Qt::AlignVCenter | Qt::AlignRight);
  QObject::connect(cancel, SIGNAL(released()), this, SLOT(emitEmpty()));

  // text box
  line = new QLineEdit();
  line->setStyleSheet(R"(
    color: black;
    background-color: white;
    font-size: 45px;
    padding: 25px;
  )");
  layout->addWidget(line, 1, 0, 1, -1);

  k = new Keyboard(this);
  QObject::connect(k, SIGNAL(emitButton(QString)), this, SLOT(getText(QString)));
  layout->addWidget(k, 2, 0, 1, -1);

  setLayout(layout);
}

void InputField::setPromptText(QString text) {
  label->setText(text);
}

void InputField::emitEmpty() {
  line->setText("");
  emit cancel();
}

void InputField::getText(QString s) {
  if (!QString::compare(s,"⌫")) {
    line->backspace();
  }

  if (!QString::compare(s,"⏎")) {
    if(line->text().length()<minTextLength){
      setPromptText("Need at least "+QString::number(minTextLength)+" characters!");
      return;
    }
    emitText(line->text());
    line->setText("");
  }

  QVector<QString> control_buttons {"⇧", "↑", "ABC", "⏎", "#+=", "⌫", "123"};
  for(QString c : control_buttons) {
    if (!QString::compare(s, c)) {
      return;
    }
  }

  line->insert(s.left(1));
}

