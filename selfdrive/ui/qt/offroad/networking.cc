#include "selfdrive/ui/qt/offroad/networking.h"

#include <algorithm>

#include <QDebug>
#include <QHBoxLayout>
#include <QLabel>
#include <QScrollBar>

#include "selfdrive/ui/qt/qt_window.h"
#include "selfdrive/ui/qt/widgets/controls.h"
#include "selfdrive/ui/qt/widgets/scrollview.h"


// Networking functions

Networking::Networking(QWidget* parent, bool show_advanced) : QFrame(parent) {
  main_layout = new QStackedLayout(this);

  wifi = new WifiManager(this);
  connect(wifi, &WifiManager::refreshSignal, this, &Networking::refresh);
  connect(wifi, &WifiManager::wrongPassword, this, &Networking::wrongPassword);

  QWidget *wifiScreen = new QWidget(this);
  QVBoxLayout* vlayout = new QVBoxLayout(wifiScreen);
  vlayout->setContentsMargins(20, 20, 20, 20);
  if (show_advanced) {
    QPushButton* advancedSettings = new QPushButton(tr("Advanced"));
    advancedSettings->setObjectName("advanced_btn");
    advancedSettings->setStyleSheet("margin-right: 30px;");
    advancedSettings->setFixedSize(400, 100);
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

  setStyleSheet(R"(
    #wifiWidget > QPushButton, #back_btn, #advanced_btn {
      font-size: 50px;
      margin: 0px;
      padding: 15px;
      border-width: 0;
      border-radius: 30px;
      color: #dddddd;
      background-color: #393939;
    }
    #back_btn:pressed, #advanced_btn:pressed {
      background-color:  #4a4a4a;
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
    QString pass = InputDialog::getText(tr("Enter password"), this, tr("for \"%1\"").arg(QString::fromUtf8(n.ssid)), true, 8);
    if (!pass.isEmpty()) {
      wifi->connect(n, pass);
    }
  }
}

void Networking::wrongPassword(const QString &ssid) {
  if (wifi->seenNetworks.contains(ssid)) {
    const Network &n = wifi->seenNetworks.value(ssid);
    QString pass = InputDialog::getText(tr("Wrong password"), this, tr("for \"%1\"").arg(QString::fromUtf8(n.ssid)), true, 8);
    if (!pass.isEmpty()) {
      wifi->connect(n, pass);
    }
  }
}

void Networking::showEvent(QShowEvent *event) {
  wifi->start();
}

void Networking::hideEvent(QHideEvent *event) {
  wifi->stop();
}

// AdvancedNetworking functions

AdvancedNetworking::AdvancedNetworking(QWidget* parent, WifiManager* wifi): QWidget(parent), wifi(wifi) {
  QVBoxLayout* main_layout = new QVBoxLayout(this);
  main_layout->setMargin(40);
  main_layout->setSpacing(20);

  // Back button
  QPushButton* back = new QPushButton(tr("Back"));
  back->setObjectName("back_btn");
  back->setFixedSize(400, 100);
  connect(back, &QPushButton::clicked, [=]() { emit backPress(); });
  main_layout->addWidget(back, 0, Qt::AlignLeft);

  ListWidget *list = new ListWidget(this);
  // Enable tethering layout
  tetheringToggle = new ToggleControl(tr("Enable Tethering"), "", "", wifi->isTetheringEnabled());
  list->addItem(tetheringToggle);
  QObject::connect(tetheringToggle, &ToggleControl::toggleFlipped, this, &AdvancedNetworking::toggleTethering);

  // Change tethering password
  ButtonControl *editPasswordButton = new ButtonControl(tr("Tethering Password"), tr("EDIT"));
  connect(editPasswordButton, &ButtonControl::clicked, [=]() {
    QString pass = InputDialog::getText(tr("Enter new tethering password"), this, "", true, 8, wifi->getTetheringPassword());
    if (!pass.isEmpty()) {
      wifi->changeTetheringPassword(pass);
    }
  });
  list->addItem(editPasswordButton);

  // IP address
  ipLabel = new LabelControl(tr("IP Address"), wifi->ipv4_address);
  list->addItem(ipLabel);

  // SSH keys
  list->addItem(new SshToggle());
  list->addItem(new SshControl());

  // Roaming toggle
  const bool roamingEnabled = params.getBool("GsmRoaming");
  ToggleControl *roamingToggle = new ToggleControl(tr("Enable Roaming"), "", "", roamingEnabled);
  QObject::connect(roamingToggle, &SshToggle::toggleFlipped, [=](bool state) {
    params.putBool("GsmRoaming", state);
    wifi->updateGsmSettings(state, QString::fromStdString(params.get("GsmApn")));
  });
  list->addItem(roamingToggle);

  // APN settings
  ButtonControl *editApnButton = new ButtonControl(tr("APN Setting"), tr("EDIT"));
  connect(editApnButton, &ButtonControl::clicked, [=]() {
    const bool roamingEnabled = params.getBool("GsmRoaming");
    const QString cur_apn = QString::fromStdString(params.get("GsmApn"));
    QString apn = InputDialog::getText(tr("Enter APN"), this, tr("leave blank for automatic configuration"), false, -1, cur_apn).trimmed();

    if (apn.isEmpty()) {
      params.remove("GsmApn");
    } else {
      params.put("GsmApn", apn.toStdString());
    }
    wifi->updateGsmSettings(roamingEnabled, apn);
  });
  list->addItem(editApnButton);

  // Set initial config
  wifi->updateGsmSettings(roamingEnabled, QString::fromStdString(params.get("GsmApn")));

  main_layout->addWidget(new ScrollView(list, this));
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

// WifiItem

WifiItem::WifiItem(QWidget *parent) : QWidget(parent) {
  QHBoxLayout *hlayout = new QHBoxLayout(this);
  hlayout->setContentsMargins(44, 0, 73, 0);
  hlayout->setSpacing(50);

  // Clickable SSID label
  ssidLabel = new ElidedLabel();
  ssidLabel->setObjectName("ssidLabel");
  ssidLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
  hlayout->addWidget(ssidLabel);

  connecting = new QPushButton(tr("CONNECTING..."));
  connecting->setObjectName("connecting");
  hlayout->addWidget(connecting, 0, Qt::AlignRight);

  forgetBtn = new QPushButton(tr("FORGET"));
  forgetBtn->setObjectName("forgetBtn");
  hlayout->addWidget(forgetBtn, 0, Qt::AlignRight);

  iconLabel = new QLabel();
  hlayout->addWidget(iconLabel, 0, Qt::AlignRight);

  strengthLabel = new QLabel();
  hlayout->addWidget(strengthLabel, 0, Qt::AlignRight);

  QObject::connect(ssidLabel, &ElidedLabel::clicked, [this]() { emit connectToNetwork(network); });
  QObject::connect(forgetBtn, &QPushButton::clicked, [this]() { emit forgotNetwork(network); });

  setVisible(false);
}

void WifiItem::update(const Network &n, const QMap<QString, QPixmap> &pixmaps, bool show_forget_btn) {
  network = n;
  ssidLabel->setText(n.ssid);
  ssidLabel->setEnabled(n.security_type != SecurityType::UNSUPPORTED);
  ssidLabel->setStyleSheet(n.connected == ConnectedType::DISCONNECTED ? "font-weight:300" : "font-weight:500");

  connecting->setVisible(n.connected == ConnectedType::CONNECTING);
  forgetBtn->setVisible(show_forget_btn);

  if (n.connected == ConnectedType::CONNECTED) {
    iconLabel->setPixmap(pixmaps["checkmark"]);
  } else if (n.security_type == SecurityType::UNSUPPORTED) {
    iconLabel->setPixmap(pixmaps["circled_slash"]);
  } else if (n.security_type == SecurityType::WPA) {
    iconLabel->setPixmap(pixmaps["lock"]);
  } else {
    iconLabel->setPixmap(QPixmap());
  }
  strengthLabel->setPixmap(pixmaps[QString("strength_%1").arg(std::clamp((int)round(n.strength / 33.), 0, 3))]);
}

// WifiUI functions

WifiUI::WifiUI(QWidget *parent, WifiManager *wifi) : QWidget(parent), wifi(wifi) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(0, 0, 0, 0);
  main_layout->setSpacing(0);

  scanning_label = new QLabel(tr("Scanning for networks..."));
  scanning_label->setStyleSheet("font-size: 65px;");
  main_layout->addWidget(scanning_label, 1, Qt::AlignCenter);

  list_container = new QWidget(this);
  QVBoxLayout *list_layout = new QVBoxLayout(list_container);
  wifi_list_widget = new ListWidget(list_container);
  list_layout->addWidget(wifi_list_widget);
  main_layout->addWidget(list_container);

  // load imgs
  std::array suffix = {"low", "medium", "high", "full"};
  for (int i = 0; i < suffix.size(); ++i) {
    auto path = ASSET_PATH + "/offroad/icon_wifi_strength_" + suffix[i] + ".svg";
    pixmaps[QString("strength_%1").arg(i)] = QPixmap(path).scaledToHeight(68, Qt::SmoothTransformation);
  }
  pixmaps["lock"] = QPixmap(ASSET_PATH + "offroad/icon_lock_closed.svg").scaledToWidth(49, Qt::SmoothTransformation);
  pixmaps["checkmark"] = QPixmap(ASSET_PATH + "offroad/icon_checkmark.svg").scaledToWidth(49, Qt::SmoothTransformation);
  pixmaps["circled_slash"] = QPixmap(ASSET_PATH + "img_circled_slash.svg").scaledToWidth(49, Qt::SmoothTransformation);

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
    #ssidLabel:disabled {
      color: #696969;
    }
  )");
}

void WifiUI::refresh() {
  bool is_empty = wifi->seenNetworks.isEmpty();
  scanning_label->setVisible(is_empty);
  list_container->setVisible(!is_empty);
  if (is_empty) return;

  const bool is_tethering_enabled = wifi->isTetheringEnabled();
  QList<Network> sortedNetworks = wifi->seenNetworks.values();
  std::sort(sortedNetworks.begin(), sortedNetworks.end(), compare_by_strength);
  int cnt = 0;
  for (const Network &network : sortedNetworks) {
    WifiItem *item = nullptr;
    if (cnt < wifi_items.size()) {
      item = wifi_items[cnt];
    } else {
      item = new WifiItem(this);
      QObject::connect(item, &WifiItem::connectToNetwork, this, &WifiUI::connectToNetwork);
      QObject::connect(item, &WifiItem::forgotNetwork, [this](const Network &n) {
        if (ConfirmationDialog::confirm(QString("Forget Wi-Fi Network \"%1\"?").arg(QString::fromUtf8(n.ssid)), this)) {
          wifi->forgetConnection(n.ssid);
        }
      });
      wifi_items.push_back(item);
      wifi_list_widget->addItem(item);
    }

    bool show_forget_btn = wifi->isKnownConnection(network.ssid) && !is_tethering_enabled;
    item->update(network, pixmaps, show_forget_btn);
    item->setVisible(true);

    ++cnt;
  }

  for (int i = cnt; i < wifi_items.size(); ++i) {
    wifi_items[i]->setVisible(false);
  }

  // repaint bottom line
  wifi_list_widget->update();
}
