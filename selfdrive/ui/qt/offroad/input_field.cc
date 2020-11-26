#include <QEvent>
#include <QVBoxLayout>
#include <QLineEdit>
#include <QLabel>
#include <QPushButton>

#include "input_field.hpp"
#include "keyboard.hpp"

InputField::InputField(QWidget *parent): QWidget(parent) {
  l = new QVBoxLayout();
  QHBoxLayout *r = new QHBoxLayout();
  label = new QLabel(this);
  label->setText("password");
  r->addWidget(label);
  QPushButton* cancel = new QPushButton("cancel");
  QObject::connect(cancel, SIGNAL(released()), this, SLOT(emitEmpty()));  
  cancel->setFixedHeight(150);
  cancel->setFixedWidth(300);
  r->addWidget(cancel);
  l->addLayout(r);
  l->addSpacing(80);

  line = new QLineEdit("");
  l->addWidget(line);
  l->addSpacing(200);

  k = new Keyboard(this);
  QObject::connect(k, SIGNAL(emitButton(QString)), this, SLOT(getText(QString)));
  l->addWidget(k);
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

