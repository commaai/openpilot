#include <QDebug>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QButtonGroup>
#include <QStackedLayout>

#include "keyboard.hpp"

KeyboardLayout::KeyboardLayout(QWidget* parent, std::vector<QVector<QString>> layout){
  QVBoxLayout* vlayout = new QVBoxLayout;
  QButtonGroup* btn_group = new QButtonGroup(this);

  QObject::connect(btn_group, SIGNAL(buttonClicked(QAbstractButton*)), parent, SLOT(handleButton(QAbstractButton*)));

  for(auto s : layout){
    QHBoxLayout *hlayout = new QHBoxLayout;
    for(QString p : s){
      QPushButton* btn = new QPushButton(p);
      btn_group->addButton(btn);
      hlayout->addSpacing(10);
      hlayout->addWidget(btn);
    }
    vlayout->addLayout(hlayout);
  }

  setLayout(vlayout);
}

Keyboard::Keyboard(QWidget *parent) : QWidget(parent) {
  main_layout = new QStackedLayout;

  // lowercase
  std::vector<QVector<QString>> lowercase = {
    {"q","w","e","r","t","y","u","i","o","p"},
    {"a","s","d","f","g","h","j","k","l"},
    {"⇧","z","x","c","v","b","n","m","⌫"},
    {"123"," ","⏎"},
  };
  main_layout->addWidget(new KeyboardLayout(this, lowercase));

  // uppercase
  std::vector<QVector<QString>> uppercase = {
    {"Q","W","E","R","T","Y","U","I","O","P"},
    {"A","S","D","F","G","H","J","K","L"},
    {"↑","Z","X","C","V","B","N","M","⌫"},
    {"123"," ","⏎"},
  };
  main_layout->addWidget(new KeyboardLayout(this, uppercase));

  // 1234567890
  std::vector<QVector<QString>> numbers = {
    {"1","2","3","4","5","6","7","8","9","0"},
    {"-","/",":",";","(",")","$","&&","@","\""},
    {"#+=",".",",","?","!","`","⌫"},
    {"ABC"," ","⏎"},
  };
  main_layout->addWidget(new KeyboardLayout(this, numbers));

  // Special characters
  std::vector<QVector<QString>> specials = {
    {"[","]","{","}","#","%","^","*","+","="},
    {"_","\\","|","~","<",">","€","£","¥","🞄"},
    {"123",".",",","?","!","`","⌫"},
    {"ABC"," ","⏎"},
  };
  main_layout->addWidget(new KeyboardLayout(this, specials));

  setLayout(main_layout);
  main_layout->setCurrentIndex(0);

  setStyleSheet(R"(
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
