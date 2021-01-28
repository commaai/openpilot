#include <QDebug>
#include <QHBoxLayout>
#include <QLabel>
#include <QPixmap>
#include <QPushButton>
#include <QLineEdit>
#include <QRandomGenerator>

#include "networking.hpp"

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

// https://stackoverflow.com/questions/478898/how-do-i-execute-a-command-and-get-the-output-of-the-command-within-c-using-po
std::string exec(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

// Networking functions

Networking::Networking(QWidget* parent) : QWidget(parent){
  try {
    wifi = new WifiManager(this);
  } catch (std::exception &e) {
    QLabel* warning = new QLabel("Network manager is inactive!");
    warning->setStyleSheet(R"(font-size: 65px;)");

    QVBoxLayout* warning_layout = new QVBoxLayout;
    warning_layout->addWidget(warning, 0, Qt::AlignCenter);
    setLayout(warning_layout);
    return;
  }
  connect(wifi, SIGNAL(wrongPassword(QString)), this, SLOT(wrongPassword(QString)));
  connect(wifi, SIGNAL(successfulConnection(QString)), this, SLOT(successfulConnection(QString)));


  s = new QStackedLayout;

  inputField = new InputField(this, 8);
  connect(inputField, SIGNAL(emitText(QString)), this, SLOT(receiveText(QString)));
  connect(inputField, SIGNAL(cancel()), this, SLOT(abortTextInput()));
  inputField->setContentsMargins(100,0,100,0);
  s->addWidget(inputField);

  QVBoxLayout* vlayout = new QVBoxLayout;
  QPushButton* advancedSettings = new QPushButton("Advanced");
  advancedSettings->setStyleSheet(R"(margin-right: 30px)");
  advancedSettings->setFixedSize(300, 100);
  connect(advancedSettings, &QPushButton::released, [=](){s->setCurrentIndex(2);});
  vlayout->addSpacing(10);
  vlayout->addWidget(advancedSettings, 0, Qt::AlignRight);
  vlayout->addSpacing(10);

  wifiWidget = new WifiUI(0, 5, wifi);
  connect(wifiWidget, SIGNAL(connectToNetwork(Network)), this, SLOT(connectToNetwork(Network)));
  vlayout->addWidget(wifiWidget, 1);

  s->addWidget(layoutToWidget(vlayout, this));

  an = new AdvancedNetworking(this, wifi);
  connect(an, &AdvancedNetworking::backPress, [=](){s->setCurrentIndex(1);});
  connect(an, &AdvancedNetworking::openKeyboard, [=](){emit openKeyboard();});
  connect(an, &AdvancedNetworking::closeKeyboard, [=](){emit closeKeyboard();});
  s->addWidget(an);

  s->setCurrentIndex(1);

  // Update network status
  QTimer* timer = new QTimer(this);
  QObject::connect(timer, SIGNAL(timeout()), this, SLOT(refresh()));
  timer->start(5000);

  setStyleSheet(R"(
    QPushButton {
      font-size: 50px;
      margin: 0px;
      padding: 15px;
      border-radius: 25px;
      color: #dddddd;
      background-color: #444444;
    }
    QPushButton:disabled {
      color: #777777;
      background-color: #222222;
    }
  )");
  setLayout(s);
}

void Networking::refresh(){
  if(!this->isVisible()){
    return;
  }
  wifiWidget->refresh();
  an->refresh();
}

void Networking::connectToNetwork(Network n) {
  if (n.security_type == SecurityType::OPEN) {
    wifi->connect(n);
  } else if (n.security_type == SecurityType::WPA) {
    inputField->setPromptText("Enter password for \"" + n.ssid + "\"");
    s->setCurrentIndex(0);
    selectedNetwork = n;
    emit openKeyboard();
  }
}

void Networking::abortTextInput(){
  s->setCurrentIndex(1);
    emit closeKeyboard();
}

void Networking::receiveText(QString text) {
  wifi->disconnect();
  wifi->connect(selectedNetwork, text);
  s->setCurrentIndex(1);
  emit closeKeyboard();
}

void Networking::wrongPassword(QString ssid) {
  if(s->currentIndex()==0){
    qDebug()<<"Wrong password, but we are already trying a new network";
    return;
  }
  if(s->currentIndex()==2){
    qDebug()<<"Wrong password, but we are in advanced settings";
    return;
  }
  if(!this->isVisible()){
    qDebug()<<"Wrong password, but we are not visible";
    return;

  }
  for (Network n : wifi->seen_networks) {
    if (n.ssid == ssid) {
      inputField->setPromptText("Wrong password for \"" + n.ssid +"\"");
      s->setCurrentIndex(0);
      emit openKeyboard();
      return;
    }
  }
}

void Networking::successfulConnection(QString ssid) {
  //Maybe we will want to do something here in the future.
}

void Networking::sidebarChange(){
  if (s == nullptr || an == nullptr){
    return;
  }

  s->setCurrentIndex(1);
  an->s->setCurrentIndex(1);
  refresh();
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
  s = new QStackedLayout;// inputField, mainPage, SSH settings
  inputField = new InputField(this, 8);
  connect(inputField, SIGNAL(emitText(QString)), this, SLOT(receiveText(QString)));
  connect(inputField, SIGNAL(cancel()), this, SLOT(abortTextInput()));
  inputField->setContentsMargins(100,0,100,0);
  s->addWidget(inputField);

  QVBoxLayout* vlayout = new QVBoxLayout;

  //Back button
  QHBoxLayout* backLayout = new QHBoxLayout;
  QPushButton* back = new QPushButton("BACK");
  back->setFixedSize(500, 100);
  connect(back, &QPushButton::released, [=](){emit backPress();});
  backLayout->addWidget(back, 0, Qt::AlignLeft);
  vlayout->addWidget(layoutToWidget(backLayout, this), 0, Qt::AlignLeft);

  //Enable tethering layout
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

  //Change tethering password
  QHBoxLayout *tetheringPassword = new QHBoxLayout;
  tetheringPassword->addWidget(new QLabel("Edit tethering password"), 1);
  editPasswordButton = new QPushButton("EDIT");
  editPasswordButton->setFixedWidth(500);
  connect(editPasswordButton, &QPushButton::released, [=](){inputField->setPromptText("Enter the new hotspot password"); s->setCurrentIndex(0); emit openKeyboard();});
  tetheringPassword->addWidget(editPasswordButton, 1, Qt::AlignRight);
  vlayout->addWidget(layoutToWidget(tetheringPassword, this), 0);
  vlayout->addWidget(hline(), 0);

  //IP adress
  QHBoxLayout* IPlayout = new QHBoxLayout;
  IPlayout->addWidget(new QLabel("IP address"), 0);
  ipLabel = new QLabel(wifi->ipv4_address);
  ipLabel->setStyleSheet("color: #aaaaaa");
  IPlayout->addWidget(ipLabel, 0, Qt::AlignRight);
  vlayout->addWidget(layoutToWidget(IPlayout, this), 0);
  vlayout->addWidget(hline(), 0);

  //Enable SSH
  QHBoxLayout* enableSSHLayout = new QHBoxLayout(this);
  enableSSHLayout->addWidget(new QLabel("Enable SSH", this));
  toggle_switch_SSH = new Toggle(this);
  toggle_switch_SSH->immediateOffset = 40;
  toggle_switch_SSH->setFixedSize(150, 100);
  if (isSSHEnabled()) {
    toggle_switch_SSH->togglePosition();
  }
  QObject::connect(toggle_switch_SSH, SIGNAL(stateChanged(int)), this, SLOT(toggleSSH(int)));
  enableSSHLayout->addWidget(toggle_switch_SSH);
  vlayout->addWidget(layoutToWidget(enableSSHLayout, this));
  vlayout->addWidget(hline(), 0);

  //Authorized SSH keys
  QHBoxLayout* authSSHLayout = new QHBoxLayout(this);
  authSSHLayout->addWidget(new QLabel("Authorized SSH keys", this));
  QPushButton* editAuthSSHButton = new QPushButton("EDIT", this);
  editAuthSSHButton->setFixedWidth(500);
  connect(editAuthSSHButton, &QPushButton::released, [=](){s->setCurrentIndex(2);});
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

  //vlayout to widget
  QWidget* settingsWidget = layoutToWidget(vlayout, this);
  settingsWidget->setStyleSheet("margin-left: 40px; margin-right: 40px;");
  s->addWidget(settingsWidget);
  s->setCurrentIndex(1);

  ssh = new SSH;
  connect(ssh, &SSH::closeSSHSettings, [=](){s->setCurrentIndex(1);});
  s->addWidget(ssh);

  setLayout(s);
}

bool AdvancedNetworking::isSSHEnabled(){
  QString response = QString::fromStdString(exec("systemctl is-active ssh"));
  return response.startsWith("active");
}

void AdvancedNetworking::refresh(){
  ipLabel->setText(wifi->ipv4_address);
  if (toggle_switch_SSH->on != isSSHEnabled()) {
    toggle_switch_SSH->togglePosition();
  }
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
  if (enable) {
    system("sudo systemctl enable ssh");
    system("sudo systemctl start ssh");
  } else {
    system("sudo systemctl stop ssh");
    system("sudo systemctl disable ssh");

  }
}
void AdvancedNetworking::receiveText(QString text){
  wifi->changeTetheringPassword(text);
  s->setCurrentIndex(1);
  emit closeKeyboard();
}

void AdvancedNetworking::abortTextInput(){
  s->setCurrentIndex(1);
  emit closeKeyboard();
}

// WifiUI functions

WifiUI::WifiUI(QWidget *parent, int page_length, WifiManager* wifi) : QWidget(parent), networks_per_page(page_length), wifi(wifi) {
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

  int i = 0;
  int countWidgets = 0;
  int button_height = static_cast<int>(this->height() / (networks_per_page + 1) * 0.6);
  for (Network &network : wifi->seen_networks) {
    QHBoxLayout *hlayout = new QHBoxLayout;
    if (page * networks_per_page <= i && i < (page + 1) * networks_per_page) {
      // SSID
      hlayout->addSpacing(50);
      QString ssid = QString::fromUtf8(network.ssid);
      if(ssid.length() > 30){
        ssid = ssid.left(30)+"…";
      }
      hlayout->addWidget(new QLabel(ssid));

      // strength indicator
      unsigned int strength_scale = network.strength / 17;
      QPixmap pix("../assets/images/network_" + QString::number(strength_scale) + ".png");
      QLabel *icon = new QLabel();
      icon->setPixmap(pix.scaledToWidth(100, Qt::SmoothTransformation));
      icon->setSizePolicy(QSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed));
      hlayout->addWidget(icon);
      hlayout->addSpacing(20);

      // connect button
      QPushButton* btn = new QPushButton(network.security_type == SecurityType::UNSUPPORTED ? "Unsupported" : (network.connected == ConnectedType::CONNECTED ? "Connected" : (network.connected == ConnectedType::CONNECTING ? "Connecting" : "Connect")));
      btn->setFixedWidth(400);
      btn->setFixedHeight(button_height);
      btn->setDisabled(network.connected == ConnectedType::CONNECTED || network.connected == ConnectedType::CONNECTING || network.security_type == SecurityType::UNSUPPORTED);
      hlayout->addWidget(btn);
      hlayout->addSpacing(20);

      connectButtons->addButton(btn, i);

      QWidget * w = new QWidget;
      w->setLayout(hlayout);
      vlayout->addWidget(w, 1);
      // Don't add the last horizontal line
      if (page * networks_per_page <= i+1 && i+1 < (page + 1) * networks_per_page && i+1 < wifi->seen_networks.size()) {
        vlayout->addWidget(hline(), 0);
      }
      countWidgets++;
    }
    i++;
  }

  // Pad vlayout to prevert oversized network widgets in case of low visible network count
  for (int i = countWidgets; i < networks_per_page; i++) {
    QWidget *w = new QWidget;
    // That we need to add w twice was determined empiricaly
    vlayout->addWidget(w, 1);
    vlayout->addWidget(w, 1);
  }

  QHBoxLayout *prev_next_buttons = new QHBoxLayout;//Adding constructor exposes the qt bug
  QPushButton* prev = new QPushButton("Previous");
  prev->setEnabled(page);
  prev->setFixedSize(400, button_height);

  QPushButton* next = new QPushButton("Next");
  next->setFixedSize(400, button_height);

  // If there are more visible networks then we can show, enable going to next page
  next->setEnabled(wifi->seen_networks.size() > (page + 1) * networks_per_page);

  QObject::connect(prev, SIGNAL(released()), this, SLOT(prevPage()));
  QObject::connect(next, SIGNAL(released()), this, SLOT(nextPage()));
  prev_next_buttons->addWidget(prev);
  prev_next_buttons->addWidget(next);

  QWidget *w = new QWidget;
  w->setLayout(prev_next_buttons);
  vlayout->addWidget(w, 1, Qt::AlignBottom);
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
