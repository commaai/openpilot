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

#include "wifi.hpp"
#include "wifiManager.hpp"
#include "input_field.hpp"

CustomConnectButton::CustomConnectButton(QString text, int iid){
    setText(text);
    id=iid;
}

void clearLayout(QLayout* layout){
  while (QLayoutItem* item = layout->takeAt(0)){
    if (QWidget* widget = item->widget()){
      widget->deleteLater();
    }
    if (QLayout* childLayout = item->layout()){
      clearLayout(childLayout);
    }
    delete item;
  }
}
QWidget* wrap(QLayout* l){
    auto widget = new QWidget();
    widget->setLayout(l);
    return widget;
}

WifiUI::WifiUI(QWidget *parent) : QWidget(parent) {
  slayout = new QStackedLayout;

  vlayout = new QVBoxLayout;
  wifi = new WifiManager;
  refresh();

  slayout->addWidget(wrap(vlayout));
  InputField *a=new InputField();
  QObject::connect(a, SIGNAL(emitText(QString)), this, SLOT(receiveText(QString)));
  slayout->addWidget(a);
  slayout->setCurrentIndex(0);
  setLayout(slayout);

  setStyleSheet(R"(
    QLabel { font-size: 40px }
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

  // TODO: implement (not) connecting with wrong password

  // Update network list
  timer = new QTimer(this);
  QObject::connect(timer, SIGNAL(timeout()), this, SLOT(refresh()));
  timer->start(1000);

  // Scan on startup
  wifi->request_scan();
}

void WifiUI::refresh(){
  if (!this->isVisible()){
    return;
  }

  wifi->request_scan();
  wifi->refreshNetworks();

  clearLayout(vlayout);
  int i=0;

  QButtonGroup* connectButtons=new QButtonGroup(this);
  QObject::connect(connectButtons, SIGNAL(buttonClicked(QAbstractButton*)), this, SLOT(handleButton(QAbstractButton*)));
  for (Network &network : wifi->seen_networks){
    QHBoxLayout *hlayout = new QHBoxLayout;
    hlayout->addWidget(new QLabel(QString::fromUtf8(network.ssid)));
    unsigned int strength_scale = std::round(network.strength / 25.0) * 25;
    QPixmap pix("../assets/offroad/indicator_wifi_" + QString::number(strength_scale) + ".png");
    QLabel *icon = new QLabel();
    icon->setPixmap(pix.scaledToWidth(100, Qt::SmoothTransformation));
    icon->setSizePolicy(QSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed));
    hlayout->addWidget(icon);
    hlayout->addSpacing(20);

    CustomConnectButton* m_button = new CustomConnectButton(network.connected ? "Connected" : "Connect",i);
    m_button->setFixedWidth(300);
    m_button->setDisabled(network.connected || network.security_type == SecurityType::UNSUPPORTED);
    connectButtons->addButton(m_button,i);

    hlayout->addWidget(m_button);
    hlayout->addSpacing(20);
    vlayout->addLayout(hlayout);
    i+=1;
  }
}

void WifiUI::handleButton(QAbstractButton* button){
  CustomConnectButton* m_button = static_cast<CustomConnectButton*>(button);
  int id = m_button->id;
  Network n = wifi->seen_networks[id];
  // qDebug() << "Clicked a button:" << id;
  // qDebug() << n.ssid;
  if(n.security_type==SecurityType::OPEN){
    wifi->connect(n);
  } else if (n.security_type==SecurityType::WPA){
    QString password = getStringFromUser();
    wifi->connect(n, password);
  } else {
    qDebug() << "Cannot determine a network's security type";
  }

}

QString WifiUI::getStringFromUser(){
  timer->stop();
  slayout->setCurrentIndex(1);

  QEventLoop loop;
  QObject::connect(this, SIGNAL(gotText()), &loop, SLOT(quit()));
  loop.exec();
  slayout->setCurrentIndex(0);

  return text;
}

void WifiUI::receiveText(QString t){
  gotText();
  qDebug()<<t;
  timer->start();
  text=t;
}
