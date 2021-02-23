#include <stdio.h>
#include <stdlib.h>
#include <curl/curl.h>

#include <QLabel>
#include <QVBoxLayout>
#include <QApplication>

#include "wifi.hpp"
#include "offroad/networking.hpp"
#include "widgets/input_field.hpp"
#include "qt_window.hpp"

void WifiSetup::finish() {
  qApp->exit();
}

WifiSetup::WifiSetup(QWidget *parent) {
  QHBoxLayout *main_layout = new QHBoxLayout();

  QPushButton *finish_btn = new QPushButton("Exit");
  finish_btn->setFixedSize(400, 200);
  main_layout->addWidget(finish_btn, 0, Qt::AlignBottom | Qt::AlignLeft);

  QObject::connect(finish_btn, SIGNAL(released()), this, SLOT(finish()));

  main_layout->addWidget(new Networking(this, true), 1);

  setLayout(main_layout);

  QObject::connect(this, SIGNAL(downloadFailed()), this, SLOT(nextPage()));

  setStyleSheet(R"(
    * {
      background-color: #292929;
      color: white;
    }
    QFrame {
      border-radius: 30px;
      background-color: #292929;
    }
    QPushButton {
      margin: 40px;
      padding: 5px;
      border-width: 0;
      border-radius: 30px;
      color: #dddddd;
      background-color: #444444;
    }
  )");
}

int main(int argc, char *argv[]) {
  QApplication a(argc, argv);
  WifiSetup setup;
  setMainWindow(&setup);
  return a.exec();
}
