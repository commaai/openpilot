#include "selfdrive/ui/qt/offroad/networking.h"

#include <QDebug>
#include <QHBoxLayout>
#include <QLabel>
#include <QPainter>

#include "selfdrive/ui/qt/widgets/scrollview.h"
#include "selfdrive/ui/qt/util.h"


QLabel *networkStrengthWidget(const unsigned int strength_) {
  QLabel *strength = new QLabel();
  QVector<QString> imgs({"low", "medium", "high", "full"});
  QPixmap pix("../assets/offroad/icon_wifi_strength_" + imgs.at(strength_) + ".svg");
  strength->setPixmap(pix.scaledToHeight(68, Qt::SmoothTransformation));
  strength->setSizePolicy(QSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed));
  strength->setStyleSheet("margin: 0px; padding-left: 48px; padding-right: 83px ");
  return strength;
}

// Networking functions

Networking::Networking(QWidget* parent, bool show_advanced) : QWidget(parent), show_advanced(show_advanced) {
  main_layout = new QStackedLayout(this);

  wifi = new WifiManager(this);
  connect(wifi, &WifiManager::wrongPassword, this, &Networking::wrongPassword);
  connect(wifi, &WifiManager::refreshSignal, this, &Networking::refresh);

  QWidget* wifiScreen = new QWidget(this);
  QVBoxLayout* vlayout = new QVBoxLayout(wifiScreen);
  vlayout->setContentsMargins(20, 20, 20, 20);
  if (show_advanced) {
    QPushButton* advancedSettings = new QPushButton("Advanced");
    advancedSettings->setObjectName("advancedBtn");
    advancedSettings->setStyleSheet("margin-right: 30px;");
    advancedSettings->setFixedSize(350, 100);
    connect(advancedSettings, &QPushButton::released, [=]() { main_layout->setCurrentWidget(an); });
    vlayout->addSpacing(10);
    vlayout->addWidget(advancedSettings, 0, Qt::AlignRight);
    vlayout->addSpacing(10);
  }

  wifiWidget = new WifiUI(this, wifi);
  wifiWidget->setObjectName("wifiWidget");
  connect(wifiWidget, &WifiUI::connectToNetwork, this, &Networking::connectToNetwork);
  vlayout->addWidget(new ScrollView(wifiWidget, this), 1);
  main_layout->addWidget(wifiScreen);

  an = new AdvancedNetworking(this, wifi);
  connect(an, &AdvancedNetworking::backPress, [=]() { main_layout->setCurrentWidget(wifiScreen); });
  main_layout->addWidget(an);

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
    * {
      background-color: #292929;
      border-radius: 10px;
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
  for (Network n : wifi->seen_networks) {
    if (n.ssid == ssid) {
      QString pass = InputDialog::getText("Wrong password", this, "for \"" + n.ssid +"\"", true, 8);
      if (!pass.isEmpty()) {
        wifi->connect(n, pass);
      }
      return;
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
  connect(back, &QPushButton::released, [=]() { emit backPress(); });
  main_layout->addWidget(back, 0, Qt::AlignLeft);

  // Enable tethering layout
  ToggleControl *tetheringToggle = new ToggleControl("Enable Tethering", "", "", wifi->isTetheringEnabled());
  main_layout->addWidget(tetheringToggle);
  QObject::connect(tetheringToggle, &ToggleControl::toggleFlipped, this, &AdvancedNetworking::toggleTethering);
  main_layout->addWidget(horizontal_line(), 0);

  // Change tethering password
  ButtonControl *editPasswordButton = new ButtonControl("Tethering Password", "EDIT");
  connect(editPasswordButton, &ButtonControl::released, [=]() {
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

  // Scan on startup
  QLabel *scanning = new QLabel("Scanning for networks");
  scanning->setStyleSheet(R"(font-size: 65px;)");
  main_layout->addWidget(scanning, 0, Qt::AlignCenter);
  main_layout->setSpacing(0);
}

void WifiUI::refresh() {
  clearLayout(main_layout);
  if (wifi->seen_networks.size() == 0) {
    QLabel *scanning = new QLabel("No networks found. Scanning...");
    scanning->setStyleSheet(R"(font-size: 65px;)");
    main_layout->addWidget(scanning, 0, Qt::AlignCenter);
    return;
  }
  main_layout->addSpacing(50);

  int i = 0;
  for (Network &network : wifi->seen_networks) {
    QHBoxLayout *hlayout = new QHBoxLayout;
    hlayout->setSpacing(0);

    QPushButton *ssid_label = new QPushButton(network.ssid);
    ssid_label->setEnabled(!(network.connected == ConnectedType::CONNECTED || network.connected == ConnectedType::CONNECTING || network.security_type == SecurityType::UNSUPPORTED));

    QString ssidStyleSheet = R"(
      font-size: 55px;
      text-align: left;
      background-color: transparent;
      border: 0px;
      padding: 50px;
      padding-left: 0px;
      padding-right: 0px;
      margin: 0px;
      margin-left: 44px;
    )";

    QObject::connect(ssid_label, &QPushButton::clicked, this, [=]() { emit connectToNetwork(network); });
    hlayout->addWidget(ssid_label, 1);
//    hlayout->addWidget(ssid_label, 1, Qt::AlignLeft);

    if (wifi->isKnownConnection(network.ssid) && !wifi->isTetheringEnabled()) {
      QPushButton *forgetBtn = new QPushButton("FORGET");
      forgetBtn->setStyleSheet(R"(
        font-size: 32px;
        font-weight: 500;
        color: #292929;
        background-color: #BDBDBD;
        border-width: 1px;
        border-radius: 5px;
        border-color: #828282;
        padding-left: 40px;
        padding-right: 40px;
        padding-bottom: 16px;
        padding-top: 16px;
      )");

      QObject::connect(forgetBtn, &QPushButton::released, [=]() {
        if (ConfirmationDialog::confirm("Are you sure you want to forget " + QString::fromUtf8(network.ssid) + "?", this)) {
          wifi->forgetConnection(network.ssid);
        }
      });

      hlayout->addWidget(forgetBtn, 0, Qt::AlignRight);
    }

    if (network.connected == ConnectedType::CONNECTED || network.connected == ConnectedType::CONNECTING) {
      ssidStyleSheet += "font-weight: 600;";

      QLabel *connectIcon = new QLabel();
      QPixmap pix(network.connected == ConnectedType::CONNECTED ? "../assets/offroad/icon_checkmark.png" : "../assets/navigation/direction_rotary.png");
      connectIcon->setPixmap(pix.scaledToHeight(49, Qt::SmoothTransformation));
      connectIcon->setSizePolicy(QSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed));
      connectIcon->setStyleSheet("margin: 0px; padding-left: 51px; padding-right: 0px ");
      hlayout->addWidget(connectIcon, 0, Qt::AlignRight);

    } else if (network.security_type == SecurityType::WPA) {
      QLabel *lockIcon = new QLabel();
      QPixmap pix("../assets/offroad/icon_lock_closed.svg");

      lockIcon->setPixmap(pix.scaledToHeight(49, Qt::SmoothTransformation));
      lockIcon->setSizePolicy(QSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed));
      lockIcon->setStyleSheet("margin: 0px; padding-left: 51px; padding-right: 0px ");
      hlayout->addWidget(lockIcon, 0, Qt::AlignRight);
    }
    ssid_label->setStyleSheet(ssidStyleSheet);

    // strength indicator
    hlayout->addWidget(networkStrengthWidget(network.strength / 26), 0, Qt::AlignRight);

    main_layout->addLayout(hlayout, 1);
    // Don't add the last horizontal line
    if (i+1 < wifi->seen_networks.size()) {
      main_layout->addWidget(horizontal_line(), 0);
    }
    i++;
  }
  main_layout->addStretch(3);
}
