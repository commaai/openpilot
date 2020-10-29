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

KeyboardLayout::KeyboardLayout(QWidget* parent, QVector<QString> layout[4]){
  QVBoxLayout* vlayout = new QVBoxLayout;
  QButtonGroup* myButtongroup = new QButtonGroup(this);
  QObject::connect(myButtongroup, SIGNAL(buttonClicked(QAbstractButton*)), parent, SLOT(handleButton(QAbstractButton*)));
  
  for(int i=0;i<4;i++){
    QVector<QString> s=layout[i];
    QHBoxLayout* hlayout = new QHBoxLayout;
    for(QString p:s){
      QPushButton* m_button = new QPushButton(p);
      myButtongroup->addButton(m_button);
      hlayout->addSpacing(10);
      hlayout->addWidget(m_button);
      // qDebug()<<"Adding button "<<p;
    }
    vlayout->addLayout(hlayout);
  }

  setLayout(vlayout);
}

Keyboard::Keyboard(QWidget *parent) : QWidget(parent) {//Constructor
  main_layout = new QStackedLayout;

  //lowercase
  QVector<QString> top {"q","w","e","r","t","y","u","i","o","p"};
  QVector<QString> mid {"a","s","d","f","g","h","j","k","l"};
  QVector<QString> bot {"⇧","z","x","c","v","b","n","m","⌫"};
  QVector<QString> footer {"123"," ","⏎"};
  QVector<QString> layout [4]= {top, mid, bot, footer};
  main_layout->addWidget(new KeyboardLayout(this, layout));

  //UPPERCASE
  QVector<QString> top2 {"Q","W","E","R","T","Y","U","I","O","P"};
  QVector<QString> mid2 {"A","S","D","F","G","H","J","K","L"};
  QVector<QString> bot2{"↑","Z","X","C","V","B","N","M","⌫"};
  QVector<QString> footer2 {"123"," ","⏎"};
  QVector<QString> layout2 [4]= {top2, mid2, bot2, footer2};
  main_layout->addWidget(new KeyboardLayout(this, layout2));

  //1234567890
  QVector<QString> top3 {"1","2","3","4","5","6","7","8","9","0"};
  QVector<QString> mid3 {"-","/",":",";","(",")","$","&&","@","\""};
  QVector<QString> bot3 {"#+=",".",",","?","!","`","⌫"};
  QVector<QString> footer3 {"ABC"," ","⏎"};
  QVector<QString> layout3 [4]= {top3, mid3, bot3, footer3};
  main_layout->addWidget(new KeyboardLayout(this, layout3));

  //Special characters
  QVector<QString> top4 {"[","]","{","}","#","%","^","*","+","="};
  QVector<QString> mid4 {"_","\\","|","~","<",">","€","£","¥","🞄"};
  QVector<QString> bot4 {"123",".",",","?","!","`","⌫"};
  QVector<QString> footer4 {"ABC"," ","⏎"};
  QVector<QString> layout4 [4]= {top4, mid4, bot4, footer4};
  main_layout->addWidget(new KeyboardLayout(this, layout4));

  setLayout(main_layout);
  main_layout->setCurrentIndex(0);

  setStyleSheet(R"(
    QLabel { font-size: 40px }
    QPushButton { font-size: 40px }
    * {
      background-color: #99777777;
    }
  )");
}

void Keyboard::handleButton(QAbstractButton* m_button){
  QString id = m_button->text();
  qDebug() << "Clicked a button:" << id;
  if(!QString::compare(m_button->text(),"↑")||!QString::compare(m_button->text(),"ABC")){
    main_layout->setCurrentIndex(0);
  }
   if(!QString::compare(m_button->text(),"⇧")){
    main_layout->setCurrentIndex(1);
  }
  if(!QString::compare(m_button->text(),"123")){
    main_layout->setCurrentIndex(2);
  }
  if(!QString::compare(m_button->text(),"#+=")){
    main_layout->setCurrentIndex(3);
  }
  emit emitButton(m_button->text());
}
