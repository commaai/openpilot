#include <QEvent>
#include <QVBoxLayout>
#include <QLineEdit>
#include <QLabel>
#include <QPushButton>

#include "input_field.hpp"
#include "keyboard.hpp"

InputField::InputField(QWidget *parent): QWidget(parent) {
  l = new QGridLayout();
  l->setSpacing(30);

  label = new QLabel(this);
  label->setStyleSheet(R"(font-size: 55px;)");
  l->addWidget(label, 0, 0, Qt::AlignVCenter | Qt::AlignLeft);

  QPushButton* cancel = new QPushButton("Cancel");
  cancel->setFixedSize(300, 150);
  cancel->setStyleSheet(R"(padding: 0;)");
  l->addWidget(cancel, 0, 1, Qt::AlignVCenter | Qt::AlignRight);
  QObject::connect(cancel, SIGNAL(released()), this, SLOT(emitEmpty()));

  // text box
  line = new QLineEdit();
  line->setStyleSheet(R"(
    color: black;
    background-color: white;
    font-size: 45px;
    padding: 25px;
  )");
  l->addWidget(line, 1, 0);
  l->setRowStretch(1, 1);

  k = new Keyboard(this);
  QObject::connect(k, SIGNAL(emitButton(QString)), this, SLOT(getText(QString)));
  l->addWidget(k, 2, 0);
  l->setRowStretch(2, 1);

  setLayout(l);
}

void InputField::emitEmpty(){
  emitText("");
  line->setText("");
}

void InputField::getText(QString s){
  if(!QString::compare(s,"⌫")){
    line->backspace();
  }

  if(!QString::compare(s,"⏎")){
    emitText(line->text());
    line->setText("");
  }

  QVector<QString> control_buttons {"⇧", "↑", "ABC", "⏎", "#+=", "⌫", "123"};
  for(QString c :control_buttons){
    if(!QString::compare(s, c)){
      return;
    }
  }
  line->insert(s.left(1));
}

