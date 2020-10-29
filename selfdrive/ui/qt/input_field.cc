#include "qt/input_field.hpp"
#include "qt/keyboard.hpp"
#include <QDebug>
#include <QListWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPixmap>
#include <QPushButton>
#include <QInputDialog>
#include <QLineEdit>
#include <QCoreApplication>
#include <QButtonGroup>
#include <QStackedLayout>
#include <QLayout>
#include <QLineEdit>
#include <QApplication>
#include <QDesktopWidget>

InputField::InputField(QWidget *parent){
  l= new QVBoxLayout();
  
  line = new QLineEdit("");
  line->installEventFilter(this);
  l->addWidget(line);

  k = new Keyboard();
  QObject::connect(k, SIGNAL(emitButton(QString)), this, SLOT(getText(QString)));

  setLayout(l);
}

bool InputField::eventFilter(QObject* object, QEvent* event){
  if(object == line && event->type() == QEvent::MouseButtonPress) {
    qDebug() << "click";
    k->setWindowFlags(Qt::FramelessWindowHint);
    QRect rec = QApplication::desktop()->screenGeometry();
    k->resize(rec.width(), k->height());
    k->move(1920, 1300); //TODO: move to the correct position for TICI

    // k->setWindowState(Qt::WindowFullScreen);
    // k->setAttribute(Qt::WA_NoSystemBackground);
    // k->setAttribute(Qt::WA_TranslucentBackground);
    k->show();
  }
  return false;
}

void InputField::getText(QString s){
  if(!QString::compare(s,"⌫")){
    line->setText(line->text().left(line->text().length()-1));
  }

  if(!QString::compare(s,"⏎")){
    k->hide();
    emitText(line->text());
  }

  QVector<QString> control_buttons {"⇧", "↑", "ABC", "⏎", "#+=", "⌫"};
  for(QString c:control_buttons){
    if(!QString::compare(s, c)){
      return;
    }
  }
  if(s.length()==1){
    line->insert(s);
  }
  qDebug() << "PACKETS ON THE WAAAAAAAAAAAAY " << s ;
}

