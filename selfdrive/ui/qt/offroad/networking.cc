#include <QDebug>
#include <QHBoxLayout>
#include <QLabel>
#include <QPixmap>
#include <QPushButton>
#include <QLineEdit>
#include <QRandomGenerator>
#include <algorithm>

#include "common/params.h"
#include "networking.hpp"
#include "util.h"

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

QWidget* layoutToWidget(QLayout* l, QWidget* parent){
  QWidget* q = new QWidget(parent);
  q->setLayout(l);
  return q;
}

// Networking functions

Networking::Networking(QWidget* parent, bool show_advanced) : QWidget(parent), show_advanced(show_advanced){
  s = new QStackedLayout;

  QLabel* warning = new QLabel("Network manager is inactive!");
  warning->setStyleSheet(R"(font-size: 65px;)");

  s->addWidget(warning);
  setLayout(s);

  QTimer* timer = new QTimer(this);
  QObject::connect(timer, SIGNAL(timeout()), this, SLOT(refresh()));
  timer->start(5000);
  attemptInitialization();
}

void Networking::attemptInitialization(){
  // Checks if network manager is active
  try {
    wifi = new WifiManager(this);
  } catch (std::exception &e) {
    return;
  }

  connect(wifi, SIGNAL(wrongPassword(QString)), this, SLOT(wrongPassword(QString)));

  QVBoxLayout* vlayout = new QVBoxLayout;

  if (show_advanced) {
    QPushButton* advancedSettings = new QPushButton("Advanced");
    advancedSettings->setStyleSheet(R"(margin-right: 30px)");
    advancedSettings->setFixedSize(350, 100);
    connect(advancedSettings, &QPushButton::released, [=](){s->setCurrentWidget(an);});
    vlayout->addSpacing(10);
    vlayout->addWidget(advancedSettings, 0, Qt::AlignRight);
    vlayout->addSpacing(10);
  }

  wifiWidget = new WifiUI(0, wifi);
  connect(wifiWidget, SIGNAL(connectToNetwork(Network)), this, SLOT(connectToNetwork(Network)));
  vlayout->addWidget(wifiWidget, 1);

  wifiScreen = layoutToWidget(vlayout, this);
  s->addWidget(wifiScreen);

  an = new AdvancedNetworking(this, wifi);
  connect(an, &AdvancedNetworking::backPress, [=](){s->setCurrentWidget(wifiScreen);});
  s->addWidget(an);

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
  s->setCurrentWidget(wifiScreen);
  ui_setup_complete = true;
}

void Networking::refresh(){
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

void Networking::connectToNetwork(Network n) {
  if (n.security_type == SecurityType::OPEN) {
    wifi->connect(n);
  } else if (n.security_type == SecurityType::WPA) {
    QString pass = InputDialog::getText("Enter password for \"" + n.ssid + "\"", 8);
    wifi->connect(n, pass);
  }
}

void Networking::wrongPassword(QString ssid) {
  for (Network n : wifi->seen_networks) {
    if (n.ssid == ssid) {
      QString pass = InputDialog::getText("Wrong password for \"" + n.ssid +"\"", 8);
      wifi->connect(n, pass);
      return;
    }
  }
}

QFrame* hline(QWidget* parent = 0){
  QFrame* line = new QFrame(parent);
  line->setFrameShape(QFrame::StyledPanel);
  line->setStyleSheet("margin-left: 40px; margin-right: 40px; border-width: 1px; border-bottom-style: solid; border-color: gray;");
  line->setFixedHeight(2);
  return line;
}

// AdvancedNetworking functions

AdvancedNetworking::AdvancedNetworking(QWidget* parent, WifiManager* wifi): QWidget(parent), wifi(wifi){
  s = new QStackedLayout; // mainPage, SSH settings

  QVBoxLayout* vlayout = new QVBoxLayout;

  // Back button
  QHBoxLayout* backLayout = new QHBoxLayout;
  QPushButton* back = new QPushButton("Back");
  back->setFixedSize(500, 100);
  connect(back, &QPushButton::released, [=](){emit backPress();});
  backLayout->addWidget(back, 0, Qt::AlignLeft);
  vlayout->addWidget(layoutToWidget(backLayout, this), 0, Qt::AlignLeft);

  // Enable tethering layout
  QHBoxLayout* tetheringToggleLayout = new QHBoxLayout;
  tetheringToggleLayout->addWidget(new QLabel("Enable tethering"));
  Toggle* toggle_switch = new Toggle;
  toggle_switch->setFixedSize(150, 100);
  tetheringToggleLayout->addWidget(toggle_switch);
  tetheringToggleLayout->addSpacing(40);
  if (wifi->tetheringEnabled()) {
    toggle_switch->togglePosition();
  }
  QObject::connect(toggle_switch, SIGNAL(stateChanged(int)), this, SLOT(toggleTethering(int)));
  vlayout->addWidget(layoutToWidget(tetheringToggleLayout, this), 0);
  vlayout->addWidget(hline(), 0);

  // Change tethering password
  QHBoxLayout *tetheringPassword = new QHBoxLayout;
  tetheringPassword->addWidget(new QLabel("Edit tethering password"), 1);
  editPasswordButton = new QPushButton("EDIT");
  editPasswordButton->setFixedWidth(500);
  connect(editPasswordButton, &QPushButton::released, [=](){
    QString pass = InputDialog::getText("Enter new tethering password", 8);
    if (pass.size()) {
      wifi->changeTetheringPassword(pass);
    }
  });
  tetheringPassword->addWidget(editPasswordButton, 1, Qt::AlignRight);
  vlayout->addWidget(layoutToWidget(tetheringPassword, this), 0);
  vlayout->addWidget(hline(), 0);

  // IP adress
  QHBoxLayout* IPlayout = new QHBoxLayout;
  IPlayout->addWidget(new QLabel("IP address"), 0);
  ipLabel = new QLabel(wifi->ipv4_address);
  ipLabel->setStyleSheet("color: #aaaaaa");
  IPlayout->addWidget(ipLabel, 0, Qt::AlignRight);
  vlayout->addWidget(layoutToWidget(IPlayout, this), 0);
  vlayout->addWidget(hline(), 0);

  // Enable SSH
  QHBoxLayout* enableSSHLayout = new QHBoxLayout(this);
  enableSSHLayout->addWidget(new QLabel("Enable SSH", this));
  toggle_switch_SSH = new Toggle(this);
  toggle_switch_SSH->setFixedSize(150, 100);
  if (isSSHEnabled()) {
    toggle_switch_SSH->togglePosition();
  }
  QObject::connect(toggle_switch_SSH, SIGNAL(stateChanged(int)), this, SLOT(toggleSSH(int)));
  enableSSHLayout->addWidget(toggle_switch_SSH);
  vlayout->addWidget(layoutToWidget(enableSSHLayout, this));
  vlayout->addWidget(hline(), 0);

  // SSH keys
  QHBoxLayout* authSSHLayout = new QHBoxLayout(this);
  authSSHLayout->addWidget(new QLabel("SSH keys", this));
  QPushButton* editAuthSSHButton = new QPushButton("EDIT", this);
  editAuthSSHButton->setFixedWidth(500);
  connect(editAuthSSHButton, &QPushButton::released, [=](){s->setCurrentWidget(ssh);});
  authSSHLayout->addWidget(editAuthSSHButton);
  vlayout->addWidget(layoutToWidget(authSSHLayout, this));
  vlayout->addSpacing(50);

  // //Disconnect or delete connections
  // QHBoxLayout* dangerZone = new QHBoxLayout(this);
  // QPushButton* disconnect = new QPushButton("Disconnect from WiFi", this);
  // dangerZone->addWidget(disconnect);
  // QPushButton* deleteAll = new QPushButton("DELETE ALL NETWORKS", this);
  // dangerZone->addWidget(deleteAll);
  // vlayout->addWidget(layoutToWidget(dangerZone, this));

  // vlayout to widget
  QWidget* settingsWidget = layoutToWidget(vlayout, this);
  settingsWidget->setStyleSheet("margin-left: 40px; margin-right: 40px;");
  s->addWidget(settingsWidget);

  ssh = new SSH;
  connect(ssh, &SSH::closeSSHSettings, [=](){s->setCurrentWidget(settingsWidget);});
  s->addWidget(ssh);

  setLayout(s);
}

bool AdvancedNetworking::isSSHEnabled(){
  return Params().get("SshEnabled") == "1";
}

void AdvancedNetworking::refresh(){
  ipLabel->setText(wifi->ipv4_address);
  if (toggle_switch_SSH->on != isSSHEnabled()) {
    toggle_switch_SSH->togglePosition();
  }
  // Qt can be lazy
  repaint();
}

void AdvancedNetworking::toggleTethering(int enable) {
  if (enable) {
    wifi->enableTethering();
  } else {
    wifi->disableTethering();
  }
  editPasswordButton->setEnabled(!enable);
}

void AdvancedNetworking::toggleSSH(int enable) {
  Params().write_db_value("SshEnabled", QString::number(enable).toStdString());
}


// WifiUI functions

WifiUI::WifiUI(QWidget *parent, WifiManager* wifi) : QWidget(parent), wifi(wifi) {
  vlayout = new QVBoxLayout;

  // Scan on startup
  QLabel *scanning = new QLabel("Scanning for networks");
  scanning->setStyleSheet(R"(font-size: 65px;)");
  vlayout->addWidget(scanning, 0, Qt::AlignCenter);
  vlayout->setSpacing(25);

  setLayout(vlayout);
  page = 0;
}

void WifiUI::refresh() {
  wifi->request_scan();
  wifi->refreshNetworks();
  clearLayout(vlayout);

  connectButtons = new QButtonGroup(this); // TODO check if this is a leak
  QObject::connect(connectButtons, SIGNAL(buttonClicked(QAbstractButton*)), this, SLOT(handleButton(QAbstractButton*)));

  int networks_per_page = height() / 180;

  int i = 0;
  int pageCount = (wifi->seen_networks.size() - 1) / networks_per_page;
  page = std::max(0, std::min(page, pageCount));
  for (Network &network : wifi->seen_networks) {
    QHBoxLayout *hlayout = new QHBoxLayout;
    if (page * networks_per_page <= i && i < (page + 1) * networks_per_page) {
      // SSID
      hlayout->addSpacing(50);
      QString ssid = QString::fromUtf8(network.ssid);
      if(ssid.length() > 20){
        ssid = ssid.left(20 - 3) + "…";
      }

      QLabel *ssid_label = new QLabel(ssid);
      ssid_label->setStyleSheet(R"(
        font-size: 55px;
      )");
      ssid_label->setFixedWidth(this->width()*0.5);
      hlayout->addWidget(ssid_label, 0, Qt::AlignLeft);

      // TODO: don't use images for this
      // strength indicator
      unsigned int strength_scale = network.strength / 17;
      QPixmap pix("../assets/images/network_" + QString::number(strength_scale) + ".png");
      QLabel *icon = new QLabel();
      icon->setPixmap(pix.scaledToWidth(100, Qt::SmoothTransformation));
      icon->setSizePolicy(QSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed));
      hlayout->addWidget(icon, 0, Qt::AlignRight);

      // connect button
      QPushButton* btn = new QPushButton(network.security_type == SecurityType::UNSUPPORTED ? "Unsupported" : (network.connected == ConnectedType::CONNECTED ? "Connected" : (network.connected == ConnectedType::CONNECTING ? "Connecting" : "Connect")));
      btn->setDisabled(network.connected == ConnectedType::CONNECTED || network.connected == ConnectedType::CONNECTING || network.security_type == SecurityType::UNSUPPORTED);
      btn->setFixedWidth(350);
      hlayout->addWidget(btn, 0, Qt::AlignRight);

      connectButtons->addButton(btn, i);

      vlayout->addLayout(hlayout, 1);
      // Don't add the last horizontal line
      if (page * networks_per_page <= i+1 && i+1 < (page + 1) * networks_per_page && i+1 < wifi->seen_networks.size()) {
        vlayout->addWidget(hline(), 0);
      }
    }
    i++;
  }
  vlayout->addStretch(3);


  // Setup buttons for pagination
  QHBoxLayout *prev_next_buttons = new QHBoxLayout;

  QPushButton* prev = new QPushButton("Previous");
  prev->setEnabled(page);
  QObject::connect(prev, SIGNAL(released()), this, SLOT(prevPage()));
  prev_next_buttons->addWidget(prev);

  QPushButton* next = new QPushButton("Next");
  next->setEnabled(wifi->seen_networks.size() > (page + 1) * networks_per_page);
  QObject::connect(next, SIGNAL(released()), this, SLOT(nextPage()));
  prev_next_buttons->addWidget(next);

  vlayout->addLayout(prev_next_buttons, 2);
}

void WifiUI::handleButton(QAbstractButton* button) {
  QPushButton* btn = static_cast<QPushButton*>(button);
  Network n = wifi->seen_networks[connectButtons->id(btn)];
  emit connectToNetwork(n);
}

void WifiUI::prevPage() {
  page--;
  refresh();
}

void WifiUI::nextPage() {
  page++;
  refresh();
}
