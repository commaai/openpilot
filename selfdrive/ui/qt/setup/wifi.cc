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

QLabel * title_label(QString text) {
  QLabel *l = new QLabel(text);
  l->setStyleSheet(R"(
    font-size: 100px;
    font-weight: 500;
  )");
  return l;
}

QWidget * WifiSetup::build_page(QString title, QWidget *content) {
  QVBoxLayout *main_layout = new QVBoxLayout();
  main_layout->setContentsMargins(50, 50, 50, 50);
  main_layout->addWidget(title_label(title), 0, Qt::AlignLeft | Qt::AlignTop);

  main_layout->addWidget(content);

  QHBoxLayout *nav_layout = new QHBoxLayout();

  QPushButton *finish_btn = new QPushButton("Finish");
  nav_layout->addWidget(finish_btn, 0, Qt::AlignBottom | Qt::AlignRight);
  QObject::connect(finish_btn, SIGNAL(released()), this, SLOT(finish()));

  main_layout->addLayout(nav_layout, 0);

  QWidget *widget = new QWidget();
  widget->setLayout(main_layout);
  return widget;
}

QWidget * WifiSetup::network_setup() {
  Networking *wifi = new Networking(this, true);
  return build_page("Connect to WiFi", wifi);
}

void WifiSetup::finish() {
  qApp->exit();
}

WifiSetup::WifiSetup(QWidget *parent) {
  addWidget(network_setup());

  QObject::connect(this, SIGNAL(downloadFailed()), this, SLOT(nextPage()));

  setStyleSheet(R"(
    * {
      font-family: Inter;
      color: white;
      background-color: black;
      font-size: 50px;
      border-radius: 20px;
    }
    QPushButton {
      padding: 50px;
      padding-right: 100px;
      padding-left: 100px;
      border: 7px solid white;
    }
  )");
}

int main(int argc, char *argv[]) {
  QApplication a(argc, argv);
  WifiSetup setup;
  setMainWindow(&setup);
  return a.exec();
}
