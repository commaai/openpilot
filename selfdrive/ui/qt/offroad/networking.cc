#include "selfdrive/ui/qt/offroad/networking.h"

#include <QDebug>
#include <QHBoxLayout>
#include <QLabel>
#include <QPainter>

#include "selfdrive/ui/qt/widgets/scrollview.h"
#include "selfdrive/ui/qt/util.h"

//QMutex mutex;

void NetworkStrengthWidget::paintEvent(QPaintEvent* event) {
  QPainter p(this);
  p.setRenderHint(QPainter::Antialiasing);
  p.setPen(Qt::NoPen);
  const QColor gray(0x54, 0x54, 0x54);
  for (int i = 0, x = 0; i < 5; ++i) {
    p.setBrush(i < strength_ ? Qt::white : gray);
    p.drawEllipse(x, 0, 15, 15);
    x += 20;
  }
}


// WifiWorker functions

//WifiWorker::WifiWorker(WifiManager* wifi) : WifiManager(wifi) {}

//void WifiWorker::run() {
//  while (true) {
//    QThread::msleep(500);
//    qDebug() << "refreshing wifi in thread!";
//    QMutexLocker locker(&mutex);
//    wifi->request_scan();
//    wifi->refreshNetworks();
//    emit update();
//  }
//}

//void WifiWorker::connectToNetwork(const Network &n) {
//  qDebug() << "HERE, WOOOO";
//  QMutexLocker locker(&mutex);
//  if (wifi->isKnownNetwork(n.ssid)) {
//    wifi->activateWifiConnection(n.ssid);
//  } else if (n.security_type == SecurityType::OPEN) {
//    wifi->connect(n);
//  } else if (n.security_type == SecurityType::WPA) {
//    QString pass = InputDialog::getText("Enter password for \"" + n.ssid + "\"", 8);
//    wifi->connect(n, pass);
//  }
//}

// Networking functions

Networking::Networking(QWidget* parent, bool show_advanced) : QWidget(parent), show_advanced(show_advanced) {
  main_layout = new QStackedLayout(this);

  QLabel* warning = new QLabel("Network manager is inactive!");
  warning->setAlignment(Qt::AlignCenter);
  warning->setStyleSheet(R"(font-size: 65px;)");

  main_layout->addWidget(warning);

//  QTimer* timer = new QTimer(this);
//  QObject::connect(timer, &QTimer::timeout, this, [=](){ emit refreshWifiManager(); });  // TODO cause a wifimanager refresh here
//  timer->start(1000);
  attemptInitialization();
}

//void Networking::wifiThread() {
//  while (true) {
//    QThread::sleep(1);
//    qDebug() << "refreshing wifi in thread!";
//    wifi->request_scan();
//    wifi->refreshNetworks();
//    emit refresh();
//  }
//}

//void Networking::handleResults(const QString &result) {
//  qDebug() << "got result:" << result;
//  for (Network n : wifi->seen_networks) {
//    qDebug() << n.ssid;
//  }
//}

void Networking::attemptInitialization() {
  // Checks if network manager is active
  try {
    wifiManager = new WifiManager();
  } catch (std::exception &e) {
    return;
  }
  wifiManager->moveToThread(&wifiThread);

  connect(wifiManager, &WifiManager::wrongPassword, this, &Networking::wrongPassword);
//  connect(this, &Networking::refreshNetworks, wifi, &WifiManager::refreshNetworks);

  QWidget* wifiScreen = new QWidget(this);
  QVBoxLayout* vlayout = new QVBoxLayout(wifiScreen);
  if (show_advanced) {
    QPushButton* advancedSettings = new QPushButton("Advanced");
    advancedSettings->setStyleSheet("margin-right: 30px;");
    advancedSettings->setFixedSize(350, 100);
    connect(advancedSettings, &QPushButton::released, [=]() { main_layout->setCurrentWidget(an); });
    vlayout->addSpacing(10);
    vlayout->addWidget(advancedSettings, 0, Qt::AlignRight);
    vlayout->addSpacing(10);
  }
  WifiManager* wifi = new WifiManager();

  wifiWidget = new WifiUI(this, wifi);
  vlayout->addWidget(new ScrollView(wifiWidget, this), 1);

  main_layout->addWidget(wifiScreen);

  an = new AdvancedNetworking(this, wifi);
  connect(an, &AdvancedNetworking::backPress, [=]() { main_layout->setCurrentWidget(wifiScreen); });
  main_layout->addWidget(an);


  // Set up and start wifi polling thread
//  WifiWorker *wifiWorker = new WifiManager();
  qRegisterMetaType<Network>("Network");
  qRegisterMetaType<QVector<Network>>("QVector<Network>");

  connect(&wifiThread, &QThread::finished, wifiManager, &QObject::deleteLater);

  connect(this, &Networking::startWifiManager, wifiManager, &WifiManager::start);
  connect(wifiManager, &WifiManager::updateNetworking, this, &Networking::refresh);
  connect(wifiManager, &WifiManager::tetheringStateChange, an, &AdvancedNetworking::tetheringStateChange);

  // Sub classes to wifi manager signals
  connect(wifiWidget, &WifiUI::connectToNetwork, wifiManager, &WifiManager::connectToNetwork);
  connect(an, &AdvancedNetworking::enableTethering, wifiManager, &WifiManager::enableTethering);
  connect(an, &AdvancedNetworking::disableTethering, wifiManager, &WifiManager::disableTethering);
//  connect(wifiManager, &WifiManager::updateAdvancedNetworking, an, &AdvancedNetworking::refresh);
//  connect(wifiWidget, &WifiUI::connectToNetwork, wifiWorker, &WifiWorker::connectToNetwork);

  wifiThread.start();
  emit startWifiManager();

  setStyleSheet(R"(
    QPushButton {
      font-size: 50px;
      margin: 0px;
      padding: 15px;
      border-width: 0;
      border-radius: 30px;
      color: #dddddd;
      background-color: #444444;
    }
    QPushButton:disabled {
      color: #777777;
      background-color: #222222;
    }
  )");
  main_layout->setCurrentWidget(wifiScreen);
  ui_setup_complete = true;
}

void Networking::refresh(const QVector<Network> seen_networks, const QString ipv4_address) {  // TODO: set up timer again to call this, sending a signal to the wifi thread to run once and reinitialize if needed
  qDebug() << "Networking::refresh()";
//  if (!this->isVisible()) {
//    return;
//  }
//  if (!ui_setup_complete) {
//    attemptInitialization();
//    if (!ui_setup_complete) {
//      return;
//    }
//  }

  // TODO: emit signal to wifi manager thread here
  wifiWidget->refresh(seen_networks);
  an->refresh(ipv4_address);
}

void Networking::wrongPassword(const QString &ssid) {
//  QMutexLocker locker(&mutex);
  for (Network n : wifiManager->seen_networks) {
    if (n.ssid == ssid) {
      QString pass = InputDialog::getText("Wrong password for \"" + n.ssid +"\"", 8);
      wifiManager->connect(n, pass);
      return;
    }
  }
}

// AdvancedNetworking functions

AdvancedNetworking::AdvancedNetworking(QWidget* parent, WifiManager* wifi): QWidget(parent), wifi(wifi) {

  QVBoxLayout* main_layout = new QVBoxLayout(this);
  main_layout->setMargin(40);
  main_layout->setSpacing(20);

  // Back button
  QPushButton* back = new QPushButton("Back");
  back->setFixedSize(500, 100);
  connect(back, &QPushButton::released, [=]() { emit backPress(); });
  main_layout->addWidget(back, 0, Qt::AlignLeft);

  // Enable tethering layout
//  ToggleControl *tetheringToggle = new ToggleControl("Enable Tethering", "", "", wifi->tetheringEnabled());
  tetheringToggle = new ToggleControl("Enable Tethering", "", "", false);
  main_layout->addWidget(tetheringToggle);
  QObject::connect(tetheringToggle, &ToggleControl::toggleFlipped, this, &AdvancedNetworking::toggleTethering);
  main_layout->addWidget(horizontal_line(), 0);

  // Change tethering password
  editPasswordButton = new ButtonControl("Tethering Password", "EDIT", "", [=]() {
    QString pass = InputDialog::getText("Enter new tethering password", 8);
    if (pass.size()) {
      wifi->changeTetheringPassword(pass);
    }
  });
  main_layout->addWidget(editPasswordButton, 0);
  main_layout->addWidget(horizontal_line(), 0);

  // IP address
  ipLabel = new LabelControl("IP Address", wifi->ipv4_address);
  main_layout->addWidget(ipLabel, 0);
  main_layout->addWidget(horizontal_line(), 0);

  // SSH keys
  main_layout->addWidget(new SshToggle());
  main_layout->addWidget(horizontal_line(), 0);
  main_layout->addWidget(new SshControl());

  main_layout->addStretch(1);
}

void AdvancedNetworking::refresh(const QString ipv4_address) {
//  QMutexLocker locker(&mutex);
  ipLabel->setText(ipv4_address);
  update();
}

void AdvancedNetworking::toggleTethering(bool enable) {
//  QMutexLocker locker(&mutex);
  tetheringToggle->setDisabled(true);
  if (enable) {
//    wifi->enableTethering();
    emit enableTethering();
  } else {
//    wifi->disableTethering();
    emit disableTethering();
  }
  editPasswordButton->setEnabled(!enable);  // TODO allow editing of password. on change, restart tethering
}

void AdvancedNetworking::tetheringStateChange() {
  tetheringToggle->setDisabled(false);  // on any state change, enable toggle button
}


// WifiUI functions

WifiUI::WifiUI(QWidget *parent, WifiManager* wifi) : QWidget(parent), wifi(wifi) {
  main_layout = new QVBoxLayout(this);

  // Scan on startup
  QLabel *scanning = new QLabel("Scanning for networks");
  scanning->setStyleSheet(R"(font-size: 65px;)");
  main_layout->addWidget(scanning, 0, Qt::AlignCenter);
  main_layout->setSpacing(25);
}

void WifiUI::refresh(const QVector<Network> _seen_networks) {
  clearLayout(main_layout);

  connectButtons = new QButtonGroup(this); // TODO check if this is a leak
  QObject::connect(connectButtons, qOverload<QAbstractButton*>(&QButtonGroup::buttonClicked), this, &WifiUI::handleButton);

//  mutex.lock();
//  QVector<Network> seen_networks = wifi->seen_networks;
//  mutex.unlock();
  seen_networks.clear();
  seen_networks = _seen_networks;

  int i = 0;
  for (const Network &network : seen_networks) {
    QHBoxLayout *hlayout = new QHBoxLayout;
    hlayout->addSpacing(50);

    QLabel *ssid_label = new QLabel(QString::fromUtf8(network.ssid));
    ssid_label->setStyleSheet("font-size: 55px;");
    hlayout->addWidget(ssid_label, 1, Qt::AlignLeft);

    // strength indicator
    unsigned int strength_scale = network.strength / 17;
    hlayout->addWidget(new NetworkStrengthWidget(strength_scale), 0, Qt::AlignRight);

    // connect button
    QPushButton* btn = new QPushButton(network.security_type == SecurityType::UNSUPPORTED ? "Unsupported" : (network.connected == ConnectedType::CONNECTED ? "Connected" : (network.connected == ConnectedType::CONNECTING ? "Connecting" : "Connect")));
    btn->setDisabled(network.connected == ConnectedType::CONNECTED || network.connected == ConnectedType::CONNECTING || network.security_type == SecurityType::UNSUPPORTED);
    btn->setFixedWidth(350);
    hlayout->addWidget(btn, 0, Qt::AlignRight);

    connectButtons->addButton(btn, i);

    main_layout->addLayout(hlayout, 1);
    // Don't add the last horizontal line
    if (i+1 < seen_networks.size()) {
      main_layout->addWidget(horizontal_line(), 0);
    }
    i++;
  }
  main_layout->addStretch(3);
}

//void WifiUI::handleButtonNew(QPu)

void WifiUI::handleButton(QAbstractButton* button) {
//  QMutexLocker locker(&mutex);
  QPushButton* btn = static_cast<QPushButton*>(button);
  btn->setDisabled(true);
  btn->setText("Connecting");
//  const Network n = wifi->seen_networks[connectButtons->id(btn)];
  const Network n = seen_networks[connectButtons->id(btn)];

  QString pass;
  if (n.security_type == SecurityType::WPA && !n.known) {
    pass = InputDialog::getText("Enter password for \"" + n.ssid + "\"", 8);
    if (pass.isEmpty()) {
      return;
    }
  }

  qDebug() << "emitting connectToNetwork!";
  emit connectToNetwork(n, pass);
}
