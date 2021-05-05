#include <stdio.h>
#include <stdlib.h>
#include <curl/curl.h>

#include <QLabel>
#include <QVBoxLayout>
#include <QApplication>

#include "wifi.h"
#include "offroad/networking.h"
#include "widgets/input.h"
#include "qt_window.h"

void WifiSetup::finish() {
  qApp->exit();
}

WifiSetup::WifiSetup(QWidget *parent) {
  QHBoxLayout *main_layout = new QHBoxLayout();

  QPushButton *finish_btn = new QPushButton("Exit");
  finish_btn->setFixedSize(400, 200);
  main_layout->addWidget(finish_btn, 0, Qt::AlignTop | Qt::AlignLeft);

  QObject::connect(finish_btn, &QPushButton::released, this, &WifiSetup::finish);

  QWidget* n = new Networking(this, true);

  // Next 5 lines to keep the same stylesheet on the networking widget
  QLayout* backgroundLayout = new QVBoxLayout();
  backgroundLayout->addWidget(n);
  QWidget* q = new QWidget();
  q->setLayout(backgroundLayout);
  q->setStyleSheet(R"(
  * {
    background-color: #292929;
  }
  )");
  main_layout->addWidget(q, 1);

  setLayout(main_layout);
  setStyleSheet(R"(
    * {
      background-color: black;
      color: white;
      font-size: 50px;
    }
    QVBoxLayout {
      padding: 20px;
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
