#include <cassert>
#include <iostream>


#include <QGuiApplication>
#include <QSurfaceFormat>
#include <QOpenGLContext>

#include "glwindow.hpp"
#include "window.hpp"

MainWindow::MainWindow(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout;
  GLWindow * glWindow = new GLWindow;
  //QPushButton * button1 = new QPushButton("Button 1", this);


  //main_layout->addWidget(button1);
  main_layout->addWidget(glWindow);
  main_layout->setMargin(0);


  setLayout(main_layout);
}
