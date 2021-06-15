#include "selfdrive/ui/qt/setup/wifi.h"

#include <curl/curl.h>

#include <cstdio>
#include <cstdlib>

#include <QApplication>
#include <QVBoxLayout>

#include "selfdrive/ui/qt/offroad/networking.h"
#include "selfdrive/ui/qt/qt_window.h"
#include "selfdrive/ui/qt/widgets/input.h"

void WifiSetup::finish() {
  qApp->exit();
}

WifiSetup::WifiSetup(QWidget *parent) : QWidget(parent) {
  QHBoxLayout *main_layout = new QHBoxLayout(this);

  QPushButton *finish_btn = new QPushButton("Exit");
  finish_btn->setFixedSize(400, 200);
  main_layout->addWidget(finish_btn, 0, Qt::AlignTop | Qt::AlignLeft);

  QObject::connect(finish_btn, &QPushButton::released, this, &WifiSetup::finish);

  QWidget* n = new Networking(this, true);

  // Next 5 lines to keep the same stylesheet on the networking widget
  QWidget* q = new QWidget();
  QLayout* backgroundLayout = new QVBoxLayout(q);
  backgroundLayout->addWidget(n);
  q->setStyleSheet(R"(
  * {
    background-color: #292929;
  }
  )");
  main_layout->addWidget(q, 1);

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
