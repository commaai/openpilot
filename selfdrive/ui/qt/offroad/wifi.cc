#include <QDebug>
#include <QHBoxLayout>
#include <QLabel>
#include <QPixmap>
#include <QPushButton>
#include <QLineEdit>

#include "wifi.hpp"
#include "widgets/toggle.hpp"

void clearLayout(QLayout* layout) {
  while (QLayoutItem* item = layout->takeAt(0)) {
    if (QWidget* widget = item->widget()) {
      widget->deleteLater();
    }
    if (QLayout* childLayout = item->layout()) {
      clearLayout(childLayout);
    }
    delete item;
  }
}

WifiUI::WifiUI(QWidget *parent, int page_length) : QWidget(parent), networks_per_page(page_length) {
  try {
    wifi = new WifiManager;
  } catch (std::exception &e) {
    QLabel* warning = new QLabel("Network manager is inactive!");
    warning->setStyleSheet(R"(font-size: 65px;)");

    QVBoxLayout* warning_layout = new QVBoxLayout;
    warning_layout->addWidget(warning, 0, Qt::AlignCenter);
    setLayout(warning_layout);
    return;
  }

  QObject::connect(wifi, SIGNAL(wrongPassword(QString)), this, SLOT(wrongPassword(QString)));

  QVBoxLayout * top_layout = new QVBoxLayout;
  top_layout->setSpacing(0);
  swidget = new QStackedWidget;

  // Networks page
  wifi_widget = new QWidget;
  QVBoxLayout* networkLayout = new QVBoxLayout;
  QHBoxLayout *tethering_field = new QHBoxLayout;
  tethering_field->addSpacing(50);

  ipv4 = new QLabel("");
  tethering_field->addWidget(ipv4);
  tethering_field->addWidget(new QLabel("Enable Tethering"));
  
  Toggle* toggle_switch = new Toggle(this);
  toggle_switch->setFixedSize(150, 100);
  tethering_field->addWidget(toggle_switch);
  if (wifi->tetheringEnabled()) {
    toggle_switch->togglePosition();
  }
  QObject::connect(toggle_switch, SIGNAL(stateChanged(int)), this, SLOT(toggleTethering(int)));

  QWidget* tetheringWidget = new QWidget;
  tetheringWidget->setLayout(tethering_field);
  tetheringWidget->setFixedHeight(150);
  networkLayout->addWidget(tetheringWidget);

  vlayout = new QVBoxLayout;
  wifi_widget->setLayout(vlayout);
  networkLayout->addWidget(wifi_widget);

  QWidget* networkWidget = new QWidget;
  networkWidget->setLayout(networkLayout);
  swidget->addWidget(networkWidget);

  // Keyboard page
  input_field = new InputField();
  QObject::connect(input_field, SIGNAL(emitText(QString)), this, SLOT(receiveText(QString)));
  swidget->addWidget(input_field);
  swidget->setCurrentIndex(0);

  top_layout->addWidget(swidget);
  setLayout(top_layout);

  // Update network list
  timer = new QTimer(this);
  QObject::connect(timer, SIGNAL(timeout()), this, SLOT(refresh()));
  timer->start(2000);

  // Scan on startup
  QLabel *scanning = new QLabel("Scanning for networks");
  scanning->setStyleSheet(R"(font-size: 65px;)");
  vlayout->addWidget(scanning, 0, Qt::AlignCenter);
  vlayout->setSpacing(25);

  wifi->request_scan();
  refresh();
  page = 0;
}

void WifiUI::refresh() {
  if (!this->isVisible()) {
    return;
  }

  wifi->request_scan();
  wifi->refreshNetworks();
  ipv4->setText(wifi->ipv4_address);
  clearLayout(vlayout);

  connectButtons = new QButtonGroup(this);
  QObject::connect(connectButtons, SIGNAL(buttonClicked(QAbstractButton*)), this, SLOT(handleButton(QAbstractButton*)));

  int i = 0;
  int countWidgets = 0;
  int button_height = static_cast<int>(this->height() / (networks_per_page + 1) * 0.6);
  for (Network &network : wifi->seen_networks) {
    QHBoxLayout *hlayout = new QHBoxLayout;
    if (page * networks_per_page <= i && i < (page + 1) * networks_per_page) {
      // SSID
      hlayout->addSpacing(50);
      hlayout->addWidget(new QLabel(QString::fromUtf8(network.ssid)));

      // strength indicator
      unsigned int strength_scale = network.strength / 17;
      QPixmap pix("../assets/images/network_" + QString::number(strength_scale) + ".png");
      QLabel *icon = new QLabel();
      icon->setPixmap(pix.scaledToWidth(100, Qt::SmoothTransformation));
      icon->setSizePolicy(QSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed));
      hlayout->addWidget(icon);
      hlayout->addSpacing(20);

      // connect button
      QPushButton* btn = new QPushButton(network.connected == ConnectedType::CONNECTED ? "Connected" : (network.connected == ConnectedType::CONNECTING ? "Connecting" : "Connect"));
      btn->setFixedWidth(300);
      btn->setFixedHeight(button_height);
      btn->setDisabled(network.connected == ConnectedType::CONNECTED || network.connected == ConnectedType::CONNECTING || network.security_type == SecurityType::UNSUPPORTED);
      hlayout->addWidget(btn);
      hlayout->addSpacing(20);

      connectButtons->addButton(btn, i);

      QWidget * w = new QWidget;
      w->setLayout(hlayout);
      vlayout->addWidget(w);
      w->setStyleSheet(R"(
        QLabel {
          font-size: 50px;
        }
        QPushButton {
          padding: 0;
          font-size: 50px;
          background-color: #114265;
        }
        QPushButton:disabled {
          background-color: #323C43;
        }
        * {
          background-color: #114265;
        }
      )");
      countWidgets++;
    }
    i++;
  }

  // Pad vlayout to prevert oversized network widgets in case of low visible network count
  for (int i = countWidgets; i < networks_per_page; i++) {
    QWidget *w = new QWidget;
    vlayout->addWidget(w);
  }

  QHBoxLayout *prev_next_buttons = new QHBoxLayout;
  QPushButton* prev = new QPushButton("Previous");
  prev->setEnabled(page);
  prev->setFixedHeight(button_height);
  QPushButton* next = new QPushButton("Next");
  next->setFixedHeight(button_height);

  // If there are more visible networks then we can show, enable going to next page
  next->setEnabled(wifi->seen_networks.size() > (page + 1) * networks_per_page);

  QObject::connect(prev, SIGNAL(released()), this, SLOT(prevPage()));
  QObject::connect(next, SIGNAL(released()), this, SLOT(nextPage()));
  prev_next_buttons->addWidget(prev);
  prev_next_buttons->addWidget(next);

  QWidget *w = new QWidget;
  w->setLayout(prev_next_buttons);
  w->setStyleSheet(R"(
    QPushButton {
      padding: 0;
      background-color: #114265;
    }
    QPushButton:disabled {
      background-color: #323C43;
    }
    * {
      background-color: #114265;
    }
  )");
  vlayout->addWidget(w);
}



void WifiUI::toggleTethering(int enable) {
  if (enable) {
    wifi->enableTethering();
  } else {
    wifi->disableTethering();
  }
}

void WifiUI::handleButton(QAbstractButton* button) {
  QPushButton* btn = static_cast<QPushButton*>(button);
  Network n = wifi->seen_networks[connectButtons->id(btn)];
  connectToNetwork(n);
}

void WifiUI::connectToNetwork(Network n) {
  timer->stop();
  if (n.security_type == SecurityType::OPEN) {
    wifi->connect(n);
  } else if (n.security_type == SecurityType::WPA) {
    input_field->setPromptText("Enter password for \"" + n.ssid + "\"");
    QString password = getStringFromUser();
    if (password.size()) {
      wifi->connect(n, password);
    }
  }
  refresh();
  timer->start();
}

QString WifiUI::getStringFromUser() {
  emit openKeyboard();
  swidget->setCurrentIndex(1);
  loop.exec();
  emit closeKeyboard();
  swidget->setCurrentIndex(0);
  return text;
}

void WifiUI::receiveText(QString t) {
  loop.quit();
  text = t;
}


void WifiUI::wrongPassword(QString ssid) {
  if (loop.isRunning()) {
    return;
  }
  for (Network n : wifi->seen_networks) {
    if (n.ssid == ssid) {
      input_field->setPromptText("Wrong password for \"" + n.ssid +"\"");
      connectToNetwork(n);
    }
  }
}

void WifiUI::prevPage() {
  page--;
  refresh();
}
void WifiUI::nextPage() {
  page++;
  refresh();
}
