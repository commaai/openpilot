#include <QEvent>
#include <QVBoxLayout>
#include <QLineEdit>

#include "input_field.hpp"
#include "keyboard.hpp"

InputField::InputField(QWidget *parent): QWidget(parent) {
  l = new QVBoxLayout();

  line = new QLineEdit("");
  l->addWidget(line);

  k = new Keyboard(this);
  QObject::connect(k, SIGNAL(emitButton(QString)), this, SLOT(getText(QString)));
  l->addWidget(k);
  setLayout(l);
}


void InputField::getText(QString s){
  if(!QString::compare(s,"⌫")){
    line->setText(line->text().left(line->text().length()-1));
  }

  if(!QString::compare(s,"⏎")){
    // k->hide();
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

