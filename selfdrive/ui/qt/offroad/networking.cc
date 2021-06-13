#include "selfdrive/ui/qt/offroad/networking.h"

#include <QDebug>
#include <QHBoxLayout>
#include <QLabel>
#include <QPainter>

#include "selfdrive/ui/qt/widgets/scrollview.h"
#include "selfdrive/ui/qt/util.h"


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

// Networking functions

Networking::Networking(QWidget* parent, bool show_advanced) : QWidget(parent), show_advanced(show_advanced) {
  main_layout = new QStackedLayout(this);

  QLabel* warning = new QLabel("Network manager is inactive!");
  warning->setAlignment(Qt::AlignCenter);
  warning->setStyleSheet(R"(font-size: 65px;)");

  main_layout->addWidget(warning);

  QTimer* timer = new QTimer(this);
  QObject::connect(timer, &QTimer::timeout, this, &Networking::refresh);
  timer->start(5000);
  attemptInitialization();
}

void Networking::attemptInitialization() {
  // Checks if network manager is active
  try {
    wifi = new WifiManager(this);
  } catch (std::exception &e) {
    return;
  }

  connect(wifi, &WifiManager::wrongPassword, this, &Networking::wrongPassword);

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

  wifiWidget = new WifiUI(this, wifi);
  connect(wifiWidget, &WifiUI::connectToNetwork, this, &Networking::connectToNetwork);
  vlayout->addWidget(new ScrollView(wifiWidget, this), 1);

  main_layout->addWidget(wifiScreen);

  an = new AdvancedNetworking(this, wifi);
  connect(an, &AdvancedNetworking::backPress, [=]() { main_layout->setCurrentWidget(wifiScreen); });
  main_layout->addWidget(an);

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

void Networking::refresh() {
  if (!this->isVisible()) {
    return;
  }
  if (!ui_setup_complete) {
    attemptInitialization();
    if (!ui_setup_complete) {
      return;
    }
  }
  wifiWidget->refresh();
  an->refresh();
}

void Networking::connectToNetwork(const Network &n) {
  if (n.security_type == SecurityType::OPEN) {
    wifi->connect(n);
  } else if (n.security_type == SecurityType::WPA) {
    QString pass = InputDialog::getText("Enter password for \"" + n.ssid + "\"", 8);
    wifi->connect(n, pass);
  }
}

void Networking::wrongPassword(const QString &ssid) {
  for (Network n : wifi->seen_networks) {
    if (n.ssid == ssid) {
      QString pass = InputDialog::getText("Wrong password for \"" + n.ssid +"\"", 8);
      wifi->connect(n, pass);
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
  ToggleControl *tetheringToggle = new ToggleControl("Enable Tethering", "", "", wifi->tetheringEnabled());
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

void AdvancedNetworking::refresh() {
  ipLabel->setText(wifi->ipv4_address);
  update();
}

void AdvancedNetworking::toggleTethering(bool enable) {
  if (enable) {
    wifi->enableTethering();
  } else {
    wifi->disableTethering();
  }
  editPasswordButton->setEnabled(!enable);
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

void WifiUI::refresh() {
  wifi->request_scan();
  wifi->refreshNetworks();
  clearLayout(main_layout);

  connectButtons = new QButtonGroup(this); // TODO check if this is a leak
  QObject::connect(connectButtons, qOverload<QAbstractButton*>(&QButtonGroup::buttonClicked), this, &WifiUI::handleButton);

  int i = 0;
  for (Network &network : wifi->seen_networks) {
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
    if (i+1 < wifi->seen_networks.size()) {
      main_layout->addWidget(horizontal_line(), 0);
    }
    i++;
  }
  main_layout->addStretch(3);
}

void WifiUI::handleButton(QAbstractButton* button) {
  QPushButton* btn = static_cast<QPushButton*>(button);
  Network n = wifi->seen_networks[connectButtons->id(btn)];
  emit connectToNetwork(n);
}
