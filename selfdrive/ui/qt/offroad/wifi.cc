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

WifiUI::WifiUI(QWidget *parent) : QWidget(parent) {
  vlayout = new QVBoxLayout;
  wifi = new WifiManager;
  refresh();
  setLayout(vlayout);

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

  // Update network list every second
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
    bool ok = false;
    QString password;

#ifdef QCOM2
    // TODO: implement touch keyboard
#else
    password = QInputDialog::getText(this, "Password for "+n.ssid, "Password", QLineEdit::Normal, "", &ok);
#endif
    if (ok){
      wifi->connect(n, password);
    }

  } else {
    qDebug() << "Cannot determine a network's security type";
  }

}
