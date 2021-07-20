#include "selfdrive/ui/qt/offroad/networking.h"

#include <algorithm>

#include <QDebug>
#include <QHBoxLayout>
#include <QLabel>
#include <QPainter>
#include <QScrollBar>

#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/qt/qt_window.h"
#include "selfdrive/ui/qt/widgets/scrollview.h"

bool compare_by_strength(const Network &a, const Network &b) {
  if (a.connected == ConnectedType::CONNECTED) return true;
  if (b.connected == ConnectedType::CONNECTED) return false;
  if (a.connected == ConnectedType::CONNECTING) return true;
  if (b.connected == ConnectedType::CONNECTING) return false;
  return a.strength > b.strength;
}

// Networking functions

Networking::Networking(QWidget* parent, bool show_advanced) : QFrame(parent) {
  main_layout = new QStackedLayout(this);

  wifi = new WifiManager(this);
  connect(wifi, &WifiManager::refreshSignal, this, &Networking::refresh);
  connect(wifi, &WifiManager::wrongPassword, this, &Networking::wrongPassword);

  QWidget* wifiScreen = new QWidget(this);
  QVBoxLayout* vlayout = new QVBoxLayout(wifiScreen);
  vlayout->setContentsMargins(20, 20, 20, 20);
  if (show_advanced) {
    QPushButton* advancedSettings = new QPushButton("Advanced");
    advancedSettings->setObjectName("advancedBtn");
    advancedSettings->setStyleSheet("margin-right: 30px;");
    advancedSettings->setFixedSize(350, 100);
    connect(advancedSettings, &QPushButton::clicked, [=]() { main_layout->setCurrentWidget(an); });
    vlayout->addSpacing(10);
    vlayout->addWidget(advancedSettings, 0, Qt::AlignRight);
    vlayout->addSpacing(10);
  }

  wifiWidget = new WifiUI(this, wifi);
  wifiWidget->setObjectName("wifiWidget");
  connect(wifiWidget, &WifiUI::connectToNetwork, this, &Networking::connectToNetwork);

  ScrollView *wifiScroller = new ScrollView(wifiWidget, this);
  wifiScroller->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
  vlayout->addWidget(wifiScroller, 1);
  main_layout->addWidget(wifiScreen);

  an = new AdvancedNetworking(this, wifi);
  connect(an, &AdvancedNetworking::backPress, [=]() { main_layout->setCurrentWidget(wifiScreen); });
  main_layout->addWidget(an);

  QPalette pal = palette();
  pal.setColor(QPalette::Background, QColor(0x29, 0x29, 0x29));
  setAutoFillBackground(true);
  setPalette(pal);

  // TODO: revisit pressed colors
  setStyleSheet(R"(
    #wifiWidget > QPushButton, #back_btn, #advancedBtn {
      font-size: 50px;
      margin: 0px;
      padding: 15px;
      border-width: 0;
      border-radius: 30px;
      color: #dddddd;
      background-color: #444444;
    }
  )");
  main_layout->setCurrentWidget(wifiScreen);
}

void Networking::refresh() {
  QElapsedTimer timer;
  timer.start();
  wifiWidget->refresh();
  double elapsed = timer.nsecsElapsed() / 1e6;

  qDebug() << "Took" << elapsed << "ms to draw" << wifi->seenNetworks.size() << "networks -" << elapsed / wifi->seenNetworks.size() << "ms/network";
  an->refresh();
}

void Networking::connectToNetwork(const Network &n) {
  if (wifi->isKnownConnection(n.ssid)) {
    wifi->activateWifiConnection(n.ssid);
  } else if (n.security_type == SecurityType::OPEN) {
    wifi->connect(n);
  } else if (n.security_type == SecurityType::WPA) {
    QString pass = InputDialog::getText("Enter password", this, "for \"" + n.ssid + "\"", true, 8);
    if (!pass.isEmpty()) {
      wifi->connect(n, pass);
    }
  }
}

void Networking::wrongPassword(const QString &ssid) {
  if (wifi->seenNetworks.contains(ssid)) {
    const Network &n = wifi->seenNetworks.value(ssid);
    QString pass = InputDialog::getText("Wrong password", this, "for \"" + n.ssid +"\"", true, 8);
    if (!pass.isEmpty()) {
      wifi->connect(n, pass);
    }
  }
}

void Networking::showEvent(QShowEvent* event) {
  // Wait to refresh to avoid delay when showing Networking widget
  QTimer::singleShot(300, this, [=]() {
    if (this->isVisible()) {
      wifi->refreshNetworks();
      refresh();
    }
  });
}

// AdvancedNetworking functions

AdvancedNetworking::AdvancedNetworking(QWidget* parent, WifiManager* wifi): QWidget(parent), wifi(wifi) {

  QVBoxLayout* main_layout = new QVBoxLayout(this);
  main_layout->setMargin(40);
  main_layout->setSpacing(20);

  // Back button
  QPushButton* back = new QPushButton("Back");
  back->setObjectName("back_btn");
  back->setFixedSize(500, 100);
  connect(back, &QPushButton::clicked, [=]() { emit backPress(); });
  main_layout->addWidget(back, 0, Qt::AlignLeft);

  // Enable tethering layout
  ToggleControl *tetheringToggle = new ToggleControl("Enable Tethering", "", "", wifi->isTetheringEnabled());
  main_layout->addWidget(tetheringToggle);
  QObject::connect(tetheringToggle, &ToggleControl::toggleFlipped, this, &AdvancedNetworking::toggleTethering);
  main_layout->addWidget(horizontal_line(), 0);

  // Change tethering password
  ButtonControl *editPasswordButton = new ButtonControl("Tethering Password", "EDIT");
  connect(editPasswordButton, &ButtonControl::clicked, [=]() {
    QString pass = InputDialog::getText("Enter new tethering password", this, "", true, 8, wifi->getTetheringPassword());
    if (!pass.isEmpty()) {
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

void AdvancedNetworking::refresh() {
  ipLabel->setText(wifi->ipv4_address);
  update();
}

void AdvancedNetworking::toggleTethering(bool enabled) {
  wifi->setTetheringEnabled(enabled);
}

// WifiUI functions

WifiUI::WifiUI(QWidget *parent, WifiManager* wifi) : QWidget(parent), wifi(wifi) {
  main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(0, 0, 0, 0);
  main_layout->setSpacing(0);

  // load imgs
  for (const auto &s : {"low", "medium", "high", "full"}) {
    QPixmap pix(ASSET_PATH + "/offroad/icon_wifi_strength_" + s + ".svg");
    strengths.push_back(pix.scaledToHeight(68, Qt::SmoothTransformation));
  }
  lock = QPixmap(ASSET_PATH + "offroad/icon_lock_closed.svg").scaledToWidth(49, Qt::SmoothTransformation);
  checkmark = QPixmap(ASSET_PATH + "offroad/icon_checkmark.svg").scaledToWidth(49, Qt::SmoothTransformation);

//  QLabel *scanning = new QLabel("Scanning for networks...");
//  scanning->setStyleSheet("font-size: 65px;");
//  main_layout->addWidget(scanning, 0, Qt::AlignCenter);

  setStyleSheet(R"(
    QScrollBar::handle:vertical {
      min-height: 0px;
      border-radius: 4px;
      background-color: #8A8A8A;
    }
    #forgetBtn {
      font-size: 32px;
      font-weight: 600;
      color: #292929;
      background-color: #BDBDBD;
      border-width: 1px solid #828282;
      border-radius: 5px;
      padding: 40px;
      padding-bottom: 16px;
      padding-top: 16px;
    }
    #connecting {
      font-size: 32px;
      font-weight: 600;
      color: white;
      border-radius: 0;
      padding: 27px;
      padding-left: 43px;
      padding-right: 43px;
      background-color: black;
    }
    #ssidLabel {
      font-size: 55px;
      text-align: left;
      border: none;
      padding-top: 50px;
      padding-bottom: 50px;
    }
  )");
}

void WifiUI::updateSsidLabel(QPushButton *ssidLabel, const Network &network) {
  ssidLabel->setEnabled(network.connected != ConnectedType::CONNECTED &&
                         network.connected != ConnectedType::CONNECTING &&
                         network.security_type != SecurityType::UNSUPPORTED);
  int weight = network.connected == ConnectedType::DISCONNECTED ? 300 : 500;
  ssidLabel->setStyleSheet(QString("font-weight: %1;").arg(weight));
}

void WifiUI::updateStatusIcon(QLabel *statusIcon, const Network &network) {
  if (network.connected == ConnectedType::CONNECTED) {
    statusIcon->setPixmap(checkmark);
  } else if (network.security_type == SecurityType::WPA) {
    statusIcon->setPixmap(lock);
  } else {
    statusIcon->clear();
  }
}

QHBoxLayout* WifiUI::buildNetworkWidget(const Network &network, bool isTetheringEnabled) {
  QHBoxLayout *hlayout = new QHBoxLayout;
  hlayout->setContentsMargins(44, 0, 73, 0);
  hlayout->setProperty("ssid", network.ssid);
  hlayout->setSpacing(50);

  // Clickable SSID label
  QPushButton *ssidLabel = new QPushButton(network.ssid);
  ssidLabel->setObjectName("ssidLabel");
  updateSsidLabel(ssidLabel, network);
  QObject::connect(ssidLabel, &QPushButton::clicked, this, [=]() { emit connectToNetwork(network); });
  hlayout->addWidget(ssidLabel, network.connected == ConnectedType::CONNECTING ? 0 : 1);

  // Connecting label
  QPushButton *connecting = new QPushButton("CONNECTING...");
  connecting->setObjectName("connecting");
  connecting->setVisible(network.connected == ConnectedType::CONNECTING);
  hlayout->addWidget(connecting, 2, Qt::AlignLeft);

  // Forget button
  QPushButton *forgetBtn = new QPushButton("FORGET");
  forgetBtn->setObjectName("forgetBtn");
  QObject::connect(forgetBtn, &QPushButton::clicked, [=]() {
    if (ConfirmationDialog::confirm("Forget WiFi Network \"" + QString::fromUtf8(network.ssid) + "\"?", this)) {
      wifi->forgetConnection(network.ssid);
    }
  });
  forgetBtn->setVisible(wifi->isKnownConnection(network.ssid) && !wifi->isTetheringEnabled());
  hlayout->addWidget(forgetBtn, 0, Qt::AlignRight);

  // Status icon
  QLabel *statusIcon = new QLabel();
  updateStatusIcon(statusIcon, network);
  hlayout->addWidget(statusIcon, 0, Qt::AlignRight);

  // Strength indicator
  QLabel *strength = new QLabel();
  strength->setPixmap(strengths[std::clamp((int)network.strength/26, 0, 3)]);
  hlayout->addWidget(strength, 0, Qt::AlignRight);
  return hlayout;
}

void WifiUI::updateNetworkWidget(QHBoxLayout *hlayout, const Network &network, bool isTetheringEnabled) {
  hlayout->setStretch(0, network.connected == ConnectedType::CONNECTING ? 0 : 1);

  // Clickable SSID label
  QPushButton *ssidLabel = qobject_cast<QPushButton*>(hlayout->itemAt(0)->widget());
  updateSsidLabel(ssidLabel, network);

  // Connecting label
  QPushButton *connecting = qobject_cast<QPushButton*>(hlayout->itemAt(1)->widget());
  connecting->setVisible(network.connected == ConnectedType::CONNECTING);

  // Forget button
  QPushButton *forgetBtn = qobject_cast<QPushButton*>(hlayout->itemAt(2)->widget());
  forgetBtn->setVisible(wifi->isKnownConnection(network.ssid) && !isTetheringEnabled);

  // Status icon
  QLabel *statusIcon = qobject_cast<QLabel*>(hlayout->itemAt(3)->widget());
  updateStatusIcon(statusIcon, network);

  // Strength indicator
  QLabel *strength = qobject_cast<QLabel*>(hlayout->itemAt(4)->widget());
  strength->setPixmap(strengths[std::clamp((int)network.strength/26, 0, 3)]);
}

void WifiUI::refresh() {
  if (wifi->seenNetworks.size() == 0) {
    QLabel *scanning = new QLabel("Scanning for networks...");
    scanning->setStyleSheet("font-size: 65px;");
    main_layout->addWidget(scanning, 0, Qt::AlignCenter);
    return;
  }
  QList<Network> sortedNetworks = wifi->seenNetworks.values();
  std::sort(sortedNetworks.begin(), sortedNetworks.end(), compare_by_strength);

  int i = 0;
  while (i < main_layout->count()) {
    const QString &networkSsid = main_layout->itemAt(i)->layout()->property("ssid").toString();
    if (!wifi->seenNetworks.contains(networkSsid)) {
      // TODO: is this the best way to remove the layout?
      QLayoutItem *item = main_layout->takeAt(i--);
      clearLayout(item->layout());
      delete item;
    }
    i++;
  }

  QVector<QString> drawnSsids;
  for (int i = 0; i < main_layout->count(); i++) {
    drawnSsids.push_back(main_layout->itemAt(i)->layout()->property("ssid").toString());
  }
  i = 0;
  const bool isTetheringEnabled = wifi->isTetheringEnabled();
  for (const Network &network : sortedNetworks) {
    if (!drawnSsids.contains(network.ssid)) {
      QHBoxLayout *hlayout = buildNetworkWidget(network, isTetheringEnabled);
      main_layout->insertLayout(i, hlayout, 1);
      qDebug() << network.ssid << "is not in drawn networks, add it!";
    } else {
      QHBoxLayout *hlayout = qobject_cast<QHBoxLayout*>(main_layout->itemAt(i)->layout());
      updateNetworkWidget(hlayout, network, isTetheringEnabled);
    }
    i++;
  }
  qDebug() << "-------";
//  main_layout->addStretch(1);
}
