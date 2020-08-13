#include <cassert>
#include "window.hpp"

Window::Window(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout;
  QHBoxLayout *layout1 = new QHBoxLayout;
  QHBoxLayout *layout2 = new QHBoxLayout;

  label1 = new QLabel("Test");
  layout1->addWidget(label1);

  button1 = new QPushButton("Button 1", this);
  button2 = new QPushButton("Button 2", this);

  layout2->addWidget(button1);
  layout2->addSpacerItem(new QSpacerItem(50, 10));
  layout2->addWidget(button2);


  main_layout->addLayout(layout1);
  main_layout->addLayout(layout2);

  setLayout(main_layout);


  QObject::connect(button1, SIGNAL(clicked()),
                   this, SLOT(handleButton()));
  QObject::connect(button2, SIGNAL(clicked()),
                   this, SLOT(handleButton()));

}

void Window::handleButton(){
  qDebug() << "Clicked!";
  label1->setText("Hi!");
}
