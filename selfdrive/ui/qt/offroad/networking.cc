#include "selfdrive/ui/qt/offroad/networking.h"

#include <algorithm>

#include <QDebug>
#include <QHBoxLayout>
#include <QLabel>
#include <QPainter>
#include <QScrollBar>

#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/qt/qt_window.h"
#include "selfdrive/ui/qt/widgets/controls.h"
#include "selfdrive/ui/qt/widgets/scrollview.h"


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
  pal.setColor(QPalette::Window, QColor(0x29, 0x29, 0x29));
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
  wifiWidget->refresh();
  an->refresh();
}

void Networking::connectToNetwork(const Network &n) {
  if (wifi->isKnownConnection(n.ssid)) {
    wifi->activateWifiConnection(n.ssid);
    wifiWidget->refresh();
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

  ListWidget *list = new ListWidget(this);
  // Enable tethering layout
  tetheringToggle = new ToggleControl("Enable Tethering", "", "", wifi->isTetheringEnabled());
  list->addItem(tetheringToggle);
  QObject::connect(tetheringToggle, &ToggleControl::toggleFlipped, this, &AdvancedNetworking::toggleTethering);

  // Change tethering password
  ButtonControl *editPasswordButton = new ButtonControl("Tethering Password", "EDIT");
  connect(editPasswordButton, &ButtonControl::clicked, [=]() {
    QString pass = InputDialog::getText("Enter new tethering password", this, "", true, 8, wifi->getTetheringPassword());
    if (!pass.isEmpty()) {
      wifi->changeTetheringPassword(pass);
    }
  });
  list->addItem(editPasswordButton);

  // IP address
  ipLabel = new LabelControl("IP Address", wifi->ipv4_address);
  list->addItem(ipLabel);

  // SSH keys
  list->addItem(new SshToggle());
  list->addItem(new SshControl());

  // Roaming toggle
  const bool roamingEnabled = params.getBool("GsmRoaming");
  wifi->setRoamingEnabled(roamingEnabled);
  ToggleControl *roamingToggle = new ToggleControl("Enable Roaming", "", "", roamingEnabled);
  QObject::connect(roamingToggle, &SshToggle::toggleFlipped, [=](bool state) {
    params.putBool("GsmRoaming", state);
    wifi->setRoamingEnabled(state);
  });
  list->addItem(roamingToggle);

  // APN settings
  const QString apn = QString::fromStdString(params.get("GsmApn"));
  wifi->setApn(apn);

  ButtonControl *editApnButton = new ButtonControl("APN settings", "EDIT");
  connect(editApnButton, &ButtonControl::clicked, [=]() {
    const QString cur_apn = QString::fromStdString(params.get("GsmApn"));
    QString apn = InputDialog::getText("Enter APN, leave empty for automatic config", this, "", false, -1, cur_apn);
    if (apn.isEmpty()) {
      params.remove("GsmApn");
    } else {
      params.put("GsmApn", apn.toStdString());
    }
    wifi->setApn(apn);
  });
  list->addItem(editApnButton);

  main_layout->addWidget(list);
  main_layout->addStretch(1);
}

void AdvancedNetworking::refresh() {
  ipLabel->setText(wifi->ipv4_address);
  tetheringToggle->setEnabled(true);
  update();
}

void AdvancedNetworking::toggleTethering(bool enabled) {
  wifi->setTetheringEnabled(enabled);
  tetheringToggle->setEnabled(false);
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
  circled_slash = QPixmap(ASSET_PATH + "img_circled_slash.svg").scaledToWidth(49, Qt::SmoothTransformation);

  QLabel *scanning = new QLabel("Scanning for networks...");
  scanning->setStyleSheet("font-size: 65px;");
  main_layout->addWidget(scanning, 0, Qt::AlignCenter);

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
      font-weight: 300;
      text-align: left;
      border: none;
      padding-top: 50px;
      padding-bottom: 50px;
    }
    #ssidLabel[disconnected=false] {
      font-weight: 500;
    }
    #ssidLabel:disabled {
      color: #696969;
    }
  )");
}

void WifiUI::refresh() {
  // TODO: don't rebuild this every time
  clearLayout(main_layout);

  if (wifi->seenNetworks.size() == 0) {
    QLabel *scanning = new QLabel("Scanning for networks...");
    scanning->setStyleSheet("font-size: 65px;");
    main_layout->addWidget(scanning, 0, Qt::AlignCenter);
    return;
  }
  QList<Network> sortedNetworks = wifi->seenNetworks.values();
  std::sort(sortedNetworks.begin(), sortedNetworks.end(), compare_by_strength);

  // add networks
  ListWidget *list = new ListWidget(this);
  for (Network &network : sortedNetworks) {
    QHBoxLayout *hlayout = new QHBoxLayout;
    hlayout->setContentsMargins(44, 0, 73, 0);
    hlayout->setSpacing(50);

    // Clickable SSID label
    ElidedLabel *ssidLabel = new ElidedLabel(network.ssid);
    ssidLabel->setObjectName("ssidLabel");
    ssidLabel->setEnabled(network.security_type != SecurityType::UNSUPPORTED);
    ssidLabel->setProperty("disconnected", network.connected == ConnectedType::DISCONNECTED);
    if (network.connected == ConnectedType::DISCONNECTED) {
      QObject::connect(ssidLabel, &ElidedLabel::clicked, this, [=]() { emit connectToNetwork(network); });
    }
    hlayout->addWidget(ssidLabel, network.connected == ConnectedType::CONNECTING ? 0 : 1);

    if (network.connected == ConnectedType::CONNECTING) {
      QPushButton *connecting = new QPushButton("CONNECTING...");
      connecting->setObjectName("connecting");
      hlayout->addWidget(connecting, 2, Qt::AlignLeft);
    }

    // Forget button
    if (wifi->isKnownConnection(network.ssid) && !wifi->isTetheringEnabled()) {
      QPushButton *forgetBtn = new QPushButton("FORGET");
      forgetBtn->setObjectName("forgetBtn");
      QObject::connect(forgetBtn, &QPushButton::clicked, [=]() {
        if (ConfirmationDialog::confirm("Forget Wi-Fi Network \"" + QString::fromUtf8(network.ssid) + "\"?", this)) {
          wifi->forgetConnection(network.ssid);
        }
      });
      hlayout->addWidget(forgetBtn, 0, Qt::AlignRight);
    }

    // Status icon
    if (network.connected == ConnectedType::CONNECTED) {
      QLabel *connectIcon = new QLabel();
      connectIcon->setPixmap(checkmark);
      hlayout->addWidget(connectIcon, 0, Qt::AlignRight);
    } else if (network.security_type == SecurityType::UNSUPPORTED) {
      QLabel *unsupportedIcon = new QLabel();
      unsupportedIcon->setPixmap(circled_slash);
      hlayout->addWidget(unsupportedIcon, 0, Qt::AlignRight);
    } else if (network.security_type == SecurityType::WPA) {
      QLabel *lockIcon = new QLabel();
      lockIcon->setPixmap(lock);
      hlayout->addWidget(lockIcon, 0, Qt::AlignRight);
    } else {
      hlayout->addSpacing(lock.width() + hlayout->spacing());
    }

    // Strength indicator
    QLabel *strength = new QLabel();
    strength->setPixmap(strengths[std::clamp((int)round(network.strength / 33.), 0, 3)]);
    hlayout->addWidget(strength, 0, Qt::AlignRight);

    list->addItem(hlayout);
  }
  main_layout->addWidget(list);
  main_layout->addStretch(1);
}
