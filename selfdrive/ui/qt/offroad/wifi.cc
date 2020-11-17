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
    QPushButton { font-size: 40px }
    * {
      background-color: #114265;
    }
  )");

  // TODO: Handle NetworkManager not running
  // TODO: Handle no wireless adapter found
  // TODO: periodically request scan
  // TODO: periodically update network list
  // TODO: implement connecting (including case with wrong password)

  qDebug() << "Running";
}
void WifiUI::refresh(){
  qDebug() << "";
  clearLayout(vlayout);

  wifi->refreshNetworks();
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

    QPushButton* m_button = new QPushButton((network.connected ? "Connected" : "Connect")+(QString(i, QChar(0))));
    m_button->setFixedWidth(250);
    m_button->setDisabled(network.connected || network.security_type < 0);
    connectButtons->addButton(m_button,i);

    hlayout->addWidget(m_button);
    hlayout->addSpacing(20);
    vlayout->addLayout(hlayout);
    i+=1;
  }
  QPushButton* refreshButton = new QPushButton("Refresh networks");
  connect(refreshButton, SIGNAL (released()), this, SLOT (refresh()));
  vlayout->addWidget(refreshButton);

}

void WifiUI::handleButton(QAbstractButton* m_button){
  int id = m_button->text().length()-7;  //7="Connect".length()
  Network n = wifi->seen_networks[id];
  qDebug() << "Clicked a button:" << id;
  qDebug() << n.ssid;
  m_button->setText("Connecting");
  m_button->setDisabled(true);
  if(n.security_type==0){
    wifi->connect(n);
  }else if(n.security_type==1){
    bool ok;
    QString password = QInputDialog::getText(this, "Password for "+n.ssid, "Password", QLineEdit::Normal, "Put_the_password_HERE", &ok);
    if(ok){
      wifi->connect(n, password);
    }else{
      qDebug() << "Connection cancelled, user not willing to provide a password.";
    }
  }else{
    qDebug() << "Cannot determine a network's security type";
  }

}



