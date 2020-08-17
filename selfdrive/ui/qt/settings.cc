#include <string>
#include <iostream>
#include <sstream>

#include "qt/settings.hpp"


SettingsWindow::SettingsWindow(QWidget *parent) : QWidget(parent) {

  QWidget *container = new QWidget(this);
  QVBoxLayout *checkbox_layout = new QVBoxLayout();

  for(int i = 0; i < 25; i++){
    QCheckBox *chk = new QCheckBox("Check Box " + QString::number(i+1));
    checkbox_layout->addWidget(chk);
    checkbox_layout->addSpacing(50);
  }
  container->setLayout(checkbox_layout);

  QScrollArea *scrollArea = new QScrollArea;
  scrollArea->setWidget(container);

  QScrollerProperties sp;
  sp.setScrollMetric(QScrollerProperties::DecelerationFactor, 2.0);

  QScroller* qs = QScroller::scroller(scrollArea);
  qs->setScrollerProperties(sp);

  QHBoxLayout *main_layout = new QHBoxLayout;
  main_layout->addWidget(scrollArea);

  QPushButton * button = new QPushButton("Close");
  main_layout->addWidget(button);

  setLayout(main_layout);

  QScroller::grabGesture(scrollArea, QScroller::LeftMouseButtonGesture);
  QObject::connect(button, SIGNAL(clicked()), this, SIGNAL(closeSettings()));
}
