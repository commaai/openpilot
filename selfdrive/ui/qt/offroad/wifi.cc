#include <QDebug>
#include <QHBoxLayout>
#include <QLabel>
#include <QPixmap>
#include <QPushButton>
#include <QLineEdit>

#include "wifi.hpp"


void clearLayout(QLayout* layout) {
  while (QLayoutItem* item = layout->takeAt(0)) {
    if (QWidget* widget = item->widget()){
      widget->deleteLater();
    }
    if (QLayout* childLayout = item->layout()) {
      clearLayout(childLayout);
    }
    delete item;
  }
}

WifiUI::WifiUI(QWidget *parent) : QWidget(parent) {
  wifi = new WifiManager;

  QVBoxLayout * top_layout = new QVBoxLayout;
  swidget = new QStackedWidget;

  // Networks page
  wifi_widget = new QWidget;
  vlayout = new QVBoxLayout;
  wifi_widget->setLayout(vlayout);
  swidget->addWidget(wifi_widget);

  // Keyboard page
  a = new InputField();
  QObject::connect(a, SIGNAL(emitText(QString)), this, SLOT(receiveText(QString)));
  swidget->addWidget(a);
  swidget->setCurrentIndex(0);

  top_layout->addWidget(swidget);
  setLayout(top_layout);
  a->setStyleSheet(R"(
    QLineEdit {
      background-color: #114265;
    }
  )");

  // TODO: implement (not) connecting with wrong password

  // Update network list
  timer = new QTimer(this);
  QObject::connect(timer, SIGNAL(timeout()), this, SLOT(refresh()));
  timer->start(400);

  // Scan on startup
  wifi->request_scan();
}

void WifiUI::refresh() {
  if (!this->isVisible()) {
    return;
  }

  wifi->request_scan();
  wifi->refreshNetworks();

  clearLayout(vlayout);

  connectButtons = new QButtonGroup(this);
  QObject::connect(connectButtons, SIGNAL(buttonClicked(QAbstractButton*)), this, SLOT(handleButton(QAbstractButton*)));

  int i = 0;
  for (Network &network : wifi->seen_networks){
    QHBoxLayout *hlayout = new QHBoxLayout;

    // SSID
    hlayout->addSpacing(50);
    hlayout->addWidget(new QLabel(QString::fromUtf8(network.ssid)));

    // strength indicator
    unsigned int strength_scale = std::round(network.strength / 25.0) * 25;
    QPixmap pix("../assets/offroad/indicator_wifi_" + QString::number(strength_scale) + ".png");
    QLabel *icon = new QLabel();
    icon->setPixmap(pix.scaledToWidth(100, Qt::SmoothTransformation));
    icon->setSizePolicy(QSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed));
    hlayout->addWidget(icon);
    hlayout->addSpacing(20);

    // connect button
    QPushButton* btn = new QPushButton(network.connected ? "Connected" : "Connect");
    btn->setFixedWidth(300);
    btn->setDisabled(network.connected || network.security_type == SecurityType::UNSUPPORTED);
    hlayout->addWidget(btn);
    hlayout->addSpacing(20);

    connectButtons->addButton(btn, i++);

    QWidget * w = new QWidget;
    w->setLayout(hlayout);
    vlayout->addWidget(w);

    w->setStyleSheet(R"(
      QLabel {
        font-size: 40px;
      }
      QPushButton:enabled {
        background-color: #114265;
      }
      QPushButton:disabled {
        background-color: #323C43;
      }
      * {
        background-color: #114265;
      }
    )");
  }
}

void WifiUI::handleButton(QAbstractButton* button) {
  QPushButton* btn = static_cast<QPushButton*>(button);
  qDebug() << connectButtons->id(btn);
  Network n = wifi->seen_networks[connectButtons->id(btn)];

  a->label->setText("Enter password for \"" + n.ssid  + "\"");

  if(n.security_type == SecurityType::OPEN){
    wifi->connect(n);
  } else if (n.security_type == SecurityType::WPA){
    QString password = getStringFromUser();

    if(password.size()){
      wifi->connect(n, password);
    }
  } else {
    qDebug() << "Cannot determine network's security type";
  }
}

QString WifiUI::getStringFromUser(){
  swidget->setCurrentIndex(1);
  loop.exec();
  swidget->setCurrentIndex(0);
  return text;
}

void WifiUI::receiveText(QString t) {
  loop.quit();
  text = t;
}
