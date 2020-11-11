#include <string>
#include <iostream>
#include <sstream>
#include <cassert>

#include "clickablelabel.hpp"
#include "settings.hpp"
#include "toggle_switch.hpp"

#include <QString>
#include <QStringList>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QPixmap>
#include <QDebug>

#include "common/params.h"

ParamsToggle::ParamsToggle(QString param, QString title, QString description, QString icon_path, QWidget *parent): QFrame(parent) , param(param) {
  QHBoxLayout *hlayout = new QHBoxLayout;
  QVBoxLayout *vlayout = new QVBoxLayout;

  hlayout->addSpacing(25);
  if (icon_path.length()){
    QPixmap pix(icon_path);
    QLabel *icon = new QLabel();
    icon->setPixmap(pix.scaledToWidth(100, Qt::SmoothTransformation));
    icon->setSizePolicy(QSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed));
    hlayout->addWidget(icon);
  } else{
    hlayout->addSpacing(100);
  }
  hlayout->addSpacing(25);

  checkbox = new QCheckBox(title);
  QLabel *label = new QLabel(description);
  label->setWordWrap(true);

  // TODO: show descriptions on tap
  //vlayout->addSpacing(50);
  vlayout->addWidget(checkbox);
  //vlayout->addWidget(label);
  //vlayout->addSpacing(50);
  hlayout->addLayout(vlayout);

  setLayout(hlayout);

  checkbox->setChecked(Params().read_db_bool(param.toStdString().c_str()));

  setStyleSheet(R"(
    QCheckBox {
      font-size: 70px;
    }
    QCheckBox::indicator {
      width: 100px;
      height: 100px;
    }
    QCheckBox::indicator:unchecked {
      image: url(../assets/offroad/circled-checkmark-empty.png);
    }
    QCheckBox::indicator:checked {
      image: url(../assets/offroad/circled-checkmark.png);
    }
    QLabel { font-size: 40px }
    * {
      background-color: #114265;
    }
  )");

  QObject::connect(checkbox, SIGNAL(stateChanged(int)), this, SLOT(checkboxClicked(int)));
}

void ParamsToggle::checkboxClicked(int state){
  char value = state ? '1': '0';
  Params().write_db_value(param.toStdString().c_str(), &value, 1);
}

QWidget * toggles_panel() {

  QVBoxLayout *toggles_list = new QVBoxLayout();
  toggles_list->setSpacing(25);

  toggles_list->addWidget(new ParamsToggle("OpenpilotEnabledToggle",
                                            "Enable Openpilot",
                                            "Use the openpilot system for adaptive cruise control and lane keep driver assistance. Your attention is required at all times to use this feature. Changing this setting takes effect when the car is powered off.",
                                            "../assets/offroad/icon_openpilot.png"
                                              ));
  toggles_list->addWidget(new ParamsToggle("LaneChangeEnabled",
                                            "Enable Lane Change Assist",
                                            "Perform assisted lane changes with openpilot by checking your surroundings for safety, activating the turn signal and gently nudging the steering wheel towards your desired lane. openpilot is not capable of checking if a lane change is safe. You must continuously observe your surroundings to use this feature.",
                                            "../assets/offroad/icon_road.png"
                                              ));
  toggles_list->addWidget(new ParamsToggle("IsLdwEnabled",
                                            "Enable Lane Departure Warnings",
                                            "Receive alerts to steer back into the lane when your vehicle drifts over a detected lane line without a turn signal activated while driving over 31mph (50kph).",
                                            "../assets/offroad/icon_warning.png"
                                              ));
  toggles_list->addWidget(new ParamsToggle("RecordFront",
                                            "Record and Upload Driver Camera",
                                            "Upload data from the driver facing camera and help improve the driver monitoring algorithm.",
                                            "../assets/offroad/icon_network.png"
                                            ));
  toggles_list->addWidget(new ParamsToggle("IsRHD",
                                            "Enable Right-Hand Drive",
                                            "Allow openpilot to obey left-hand traffic conventions and perform driver monitoring on right driver seat.",
                                            "../assets/offroad/icon_openpilot_mirrored.png"
                                            ));
  toggles_list->addWidget(new ParamsToggle("IsMetric",
                                            "Use Metric System",
                                            "Display speed in km/h instead of mp/h.",
                                            "../assets/offroad/icon_metric.png"
                                            ));
  toggles_list->addWidget(new ParamsToggle("CommunityFeaturesToggle",
                                            "Enable Community Features",
                                            "Use features from the open source community that are not maintained or supported by comma.ai and have not been confirmed to meet the standard safety model. These features include community supported cars and community supported hardware. Be extra cautious when using these features",
                                            "../assets/offroad/icon_shell.png"
                                            ));

  QWidget *widget = new QWidget;
  widget->setLayout(toggles_list);
  return widget;
}

QWidget * device_panel() {

  QVBoxLayout *device_layout = new QVBoxLayout;
  device_layout->setSpacing(50);

  Params params = Params();
  std::vector<std::pair<std::string, std::string>> labels = {
    {"Dongle ID", params.get("DongleId", false)},
    //{"Serial Number", "abcdefghijk"},
  };

  for (auto l : labels) {
    QString text = QString::fromStdString(l.first + ": " + l.second);
    device_layout->addWidget(new QLabel(text));
  }

  QPushButton *clear_cal_btn = new QPushButton("Reset Calibration");
  device_layout->addWidget(clear_cal_btn);
  QObject::connect(clear_cal_btn, &QPushButton::released, [=]() {
    Params().delete_db_value("CalibrationParams");
  });

  std::map<std::string, const char *> power_btns = {
    {"Power Off", "sudo poweroff"},
    {"Reboot", "sudo reboot"},
  };

  for (auto b : power_btns) {
    QPushButton *btn = new QPushButton(QString::fromStdString(b.first));
    device_layout->addWidget(btn);
#ifdef __aarch64__
    QObject::connect(btn, &QPushButton::released,
                     [=]() {std::system(b.second);});
#endif
  }

  QWidget *widget = new QWidget;
  widget->setLayout(device_layout);
  widget->setStyleSheet(R"(
    QPushButton {
      padding: 60px;
    }
  )");
  return widget;
}

QWidget * developer_panel() {
  QVBoxLayout *developer_layout = new QVBoxLayout;

  // TODO: enable SSH toggle and github keys

  Params params = Params();
  std::string brand = params.read_db_bool("Passive") ? "dashcam" : "openpilot";
  std::vector<std::pair<std::string, std::string>> labels = {
    {"Version", brand + " v" + params.get("Version", false)},
    {"Git Branch", params.get("GitBranch", false)},
    {"Git Commit", params.get("GitCommit", false).substr(0, 10)},
    {"Panda Firmware", params.get("PandaFirmwareHex", false)},
  };

  for (auto l : labels) {
    QString text = QString::fromStdString(l.first + ": " + l.second);
    developer_layout->addWidget(new QLabel(text));
  }

  QWidget *widget = new QWidget;
  widget->setLayout(developer_layout);
  return widget;
}

void SettingsWindow::setActivePanel() {
  QPushButton *btn = qobject_cast<QPushButton*>(sender());
  panel_layout->setCurrentWidget(panels[btn->text()]);
}

void SettingsWindow::selected(int index) {
  Q_ASSERT(index >= 0);
  qDebug() << "panel:" << index;
  panel_layout->setCurrentIndex(index);
}

inline QWidget* SettingsWindow::newLinebreakWidget() {
  QWidget *line_break = new QWidget();
  line_break->setFixedHeight(2);
  line_break->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);

  // Sigh... line break won't obey the padding so we need to give it a margin.
  line_break->setStyleSheet(R"(
        background-color: #414141;
        margin-left: 30px;
      )");
  return line_break;
}

inline QWidget* SettingsWindow::newSettingsPanelBaseWidget() {
  auto base_widget = new QWidget();
  base_widget->setStyleSheet(R"(
    * {
      background-color: #292929;
      font-size: 30px;
      padding-left: 30px;
      padding-right: 30px;
      padding-top: 10px;
    }

    QPushButton {
      background-color: #393939;
      font-size: 20px;
      padding-bottom: 12px;
      border-radius: 20px;
    }
    QPushButton:hover {
      background-color:  #4d4d4d;
    }
  )");

  return base_widget;
}

void SettingsWindow::initGeneralSettingsWidget() {
  general_settings_widget = newSettingsPanelBaseWidget();

  QVBoxLayout *general_settings_layout = new QVBoxLayout();

  {
    auto text = new QLabel("General");
    text->setStyleSheet(R"(
      font-size: 50px;
      font-weight: bold;
    )");
    QSizePolicy text_policy(QSizePolicy::Preferred, QSizePolicy::Preferred);
    text_policy.setHorizontalStretch(8);
    text->setSizePolicy(text_policy);

    general_settings_layout->addStretch(1);
    general_settings_layout->addWidget(text);
    general_settings_layout->addStretch(2);
  }

  QStringList general_text_items = {
    "Enable openpilot",
    "Lane change assist",
    "Lane departure warnings",
    "Upload driver camera",
  };

  for (int i = 0; i < general_text_items.size(); i++) {
    auto &text = general_text_items[i];
    QHBoxLayout *row_layout = new QHBoxLayout();

    auto label = new QLabel(text);
    row_layout->addWidget(label);
    auto toggle_switch = new Switch();
    row_layout->addWidget(toggle_switch);
    general_settings_layout->addLayout(row_layout);

    if (i == general_text_items.size() - 1) continue;

    general_settings_layout->addStretch(1);
    general_settings_layout->addWidget(newLinebreakWidget());
    general_settings_layout->addStretch(1);
  }

  general_settings_layout->addStretch(general_text_items.size());
  general_settings_widget->setLayout(general_settings_layout);
}

void SettingsWindow::initDeviceSettingsWidget() {
  device_settings_widget = newSettingsPanelBaseWidget();

  QVBoxLayout *device_settings_layout = new QVBoxLayout();
  auto text = new QLabel("Device");
  text->setStyleSheet(R"(
      font-size: 50px;
      font-weight: bold;
    )");
  device_settings_layout->addStretch(1);
  device_settings_layout->addWidget(text);
  device_settings_layout->addStretch(2);

  // Camera calibration
  {
    QHBoxLayout *row_layout = new QHBoxLayout();
    auto label = new QLabel("Camera calibration");
    QSizePolicy text_policy(QSizePolicy::Preferred, QSizePolicy::Preferred);
    text_policy.setHorizontalStretch(6);
    label->setSizePolicy(text_policy);
    row_layout->addWidget(label);

    QSizePolicy btn_policy(QSizePolicy::Preferred, QSizePolicy::Preferred);
    btn_policy.setHorizontalStretch(1);
    auto btn = new QPushButton("RESET");
    btn->setSizePolicy(btn_policy);
    row_layout->addWidget(btn);
    device_settings_layout->addLayout(row_layout);
    device_settings_layout->addStretch(1);
    device_settings_layout->addWidget(newLinebreakWidget());
    device_settings_layout->addStretch(1);
  }

  // Driver Camera View
  {
    QHBoxLayout *row_layout = new QHBoxLayout();
    auto label = new QLabel("Driver camera view");
    QSizePolicy text_policy(QSizePolicy::Preferred, QSizePolicy::Preferred);
    text_policy.setHorizontalStretch(6);
    label->setSizePolicy(text_policy);
    row_layout->addWidget(label);

    QSizePolicy btn_policy(QSizePolicy::Preferred, QSizePolicy::Preferred);
    btn_policy.setHorizontalStretch(1);
    auto btn = new QPushButton("PREVIEW");
    btn->setSizePolicy(btn_policy);
    row_layout->addWidget(btn);
    device_settings_layout->addLayout(row_layout);
    device_settings_layout->addStretch(1);
    device_settings_layout->addWidget(newLinebreakWidget());
    device_settings_layout->addStretch(1);
  }

  // Comma Connect (TODO: need the checkmark icon)
  {
    QHBoxLayout *row_layout = new QHBoxLayout();
    auto label = new QLabel("Comma Connect");
    QSizePolicy text_policy(QSizePolicy::Preferred, QSizePolicy::Preferred);
    text_policy.setHorizontalStretch(9);
    label->setSizePolicy(text_policy);
    row_layout->addWidget(label);

    QSizePolicy btn_policy(QSizePolicy::Preferred, QSizePolicy::Preferred);
    btn_policy.setHorizontalStretch(1);
    auto btn = new QPushButton("+");
    btn->setEnabled(false);
    btn->setSizePolicy(btn_policy);
    row_layout->addWidget(btn);
    device_settings_layout->addLayout(row_layout);
    device_settings_layout->addStretch(1);
    device_settings_layout->addWidget(newLinebreakWidget());
    device_settings_layout->addStretch(1);
  }

  // Dongle ID (TODO: get rid of placeholder text)
  {
    QHBoxLayout *row_layout = new QHBoxLayout();
    auto label = new QLabel("Dongle ID");
    QSizePolicy text_policy(QSizePolicy::Preferred, QSizePolicy::Preferred);
    text_policy.setHorizontalStretch(5);
    label->setSizePolicy(text_policy);
    row_layout->addWidget(label);

    QSizePolicy dongle_policy(QSizePolicy::Preferred, QSizePolicy::Preferred);
    dongle_policy.setHorizontalStretch(1);
    auto dongle_label = new QLabel("ea79584g45a1bbca1");
    dongle_label->setStyleSheet(R"(
      color: #b5b5b5;
    )");
    dongle_label->setSizePolicy(dongle_policy);
    row_layout->addWidget(dongle_label);
    device_settings_layout->addLayout(row_layout);
  }

  device_settings_layout->addStretch(4);
  device_settings_widget->setLayout(device_settings_layout);
}

void SettingsWindow::initNetworkSettingsWidget() {
  network_settings_widget = newSettingsPanelBaseWidget();

  QVBoxLayout *network_settings_layout = new QVBoxLayout();
  auto text = new QLabel("Network");
  text->setStyleSheet(R"(
      font-size: 50px;
      font-weight: bold;
    )");
  network_settings_layout->addStretch(1);
  network_settings_layout->addWidget(text);
  network_settings_layout->addStretch(2);

  // WiFi settings
  {
    QHBoxLayout *row_layout = new QHBoxLayout();
    auto label = new QLabel("WiFi settings");
    QSizePolicy text_policy(QSizePolicy::Preferred, QSizePolicy::Preferred);
    text_policy.setHorizontalStretch(6);
    label->setSizePolicy(text_policy);
    row_layout->addWidget(label);

    QSizePolicy btn_policy(QSizePolicy::Preferred, QSizePolicy::Preferred);
    btn_policy.setHorizontalStretch(1);
    auto btn = new QPushButton("OPEN");
    btn->setSizePolicy(btn_policy);
    row_layout->addWidget(btn);
    network_settings_layout->addLayout(row_layout);
    network_settings_layout->addStretch(1);
    network_settings_layout->addWidget(newLinebreakWidget());
    network_settings_layout->addStretch(1);
  }

  // Tethering settings
  {
    QHBoxLayout *row_layout = new QHBoxLayout();
    auto label = new QLabel("Tethering settings");
    QSizePolicy text_policy(QSizePolicy::Preferred, QSizePolicy::Preferred);
    text_policy.setHorizontalStretch(6);
    label->setSizePolicy(text_policy);
    row_layout->addWidget(label);

    QSizePolicy btn_policy(QSizePolicy::Preferred, QSizePolicy::Preferred);
    btn_policy.setHorizontalStretch(1);
    auto btn = new QPushButton("OPEN");
    btn->setSizePolicy(btn_policy);
    row_layout->addWidget(btn);
    network_settings_layout->addLayout(row_layout);
    network_settings_layout->addStretch(1);
    network_settings_layout->addWidget(newLinebreakWidget());
    network_settings_layout->addStretch(1);
  }
  network_settings_layout->addStretch(8);
  network_settings_widget->setLayout(network_settings_layout);
}

SettingsWindow::SettingsWindow(QWidget *parent) : QWidget(parent) {

  QWidget *lWidget = new QWidget();
  QWidget *rWidget = new QWidget();
  QSizePolicy leftPolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
  leftPolicy.setHorizontalStretch(1);
  lWidget->setSizePolicy(leftPolicy);
  lWidget->setStyleSheet(R"(
    * {
      font-size: 50px;
      padding: 0;
      margin: 0;
    }
    QLabel {
      font-weight: bold;
      padding-left: 30px;
      padding-top: 10px;
    }
  )");

  // sidebar
  QVBoxLayout *left_panel_layout = new QVBoxLayout();
  left_panel_layout->setContentsMargins(0, 0, 0, 0);

  left_panel_layout->addStretch(1);
  {
    QLabel *settingsLabel = new QLabel();
    settingsLabel->setText("SETTINGS");
    settingsLabel->setStyleSheet(R"(
      padding-left: 40px;
      font-size: 20px;
      font-weight: normal;
      margin-bottom: 25px;
      color: #bfbfbf;
    )");
    left_panel_layout->addWidget(settingsLabel);
  }

  panel_layout = new QStackedLayout();

  initGeneralSettingsWidget();
  int general_panel_idx = panel_layout->addWidget(general_settings_widget);
  general_settings_label = new ClickableLabel(lWidget, general_panel_idx);
  general_settings_label->setText("General");
  left_panel_layout->addWidget(general_settings_label);

  initDeviceSettingsWidget();
  int device_panel_idx = panel_layout->addWidget(device_settings_widget);
  device_settings_label = new ClickableLabel(lWidget, device_panel_idx);
  device_settings_label->setText("Device");
  left_panel_layout->addWidget(device_settings_label);

  initNetworkSettingsWidget();
  int network_panel_idx = panel_layout->addWidget(network_settings_widget);
  network_settings_label = new ClickableLabel(lWidget, network_panel_idx);
  network_settings_label->setText("Network");
  left_panel_layout->addWidget(network_settings_label);

  developer_settings_label = new ClickableLabel(lWidget);
  developer_settings_label->setText("Developer");
  left_panel_layout->addWidget(developer_settings_label);

  left_panel_layout->addStretch(3);
  lWidget->setLayout(left_panel_layout);

  QSizePolicy rightPolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
  rightPolicy.setHorizontalStretch(3);
  rWidget->setSizePolicy(rightPolicy);
  rWidget->setLayout(panel_layout);

  QHBoxLayout *panel_separator_layout = new QHBoxLayout();

  panel_separator_layout->addWidget(lWidget);
  panel_separator_layout->addWidget(rWidget);

  setLayout(panel_separator_layout);

  // Layout switching signals
  QObject::connect(general_settings_label, SIGNAL(selected(int)), this, SLOT(selected(int)));
  QObject::connect(device_settings_label, SIGNAL(selected(int)), this, SLOT(selected(int)));
  QObject::connect(network_settings_label, SIGNAL(selected(int)), this, SLOT(selected(int)));
  QObject::connect(developer_settings_label, SIGNAL(selected(int)), this, SLOT(selected(int)));

  // close button
  //QPushButton *close_button = new QPushButton("X");
  //close_button->setStyleSheet(R"(
  //  QPushButton {
  //    padding: 50px;
  //    font-weight: bold;
  //    font-size: 100px;
  //  }
  //)");
  //sidebar_layout->addWidget(close_button);
  //QObject::connect(close_button, SIGNAL(released()), this, SIGNAL(closeSettings()));

  // setup panels
  //panels = {
  //  {"device", device_panel()},
  //  {"toggles", toggles_panel()},
  //  {"developer", developer_panel()},
  //};

  //for (auto &panel : panels) {
  //  QPushButton *btn = new QPushButton(panel.first);
  //  btn->setStyleSheet(R"(
  //    QPushButton {
  //      padding-top: 35px;
  //      padding-bottom: 35px;
  //      font-size: 60px;
  //      text-align: right;
  //      border: none;
  //      background: none;
  //      font-weight: bold;
  //    }
  //  )");

  //  sidebar_layout->addWidget(btn);
  //  panel_layout->addWidget(panel.second);
  //  QObject::connect(btn, SIGNAL(released()), this, SLOT(setActivePanel()));
  //}

  //QHBoxLayout *settings_layout = new QHBoxLayout();
  //settings_layout->addSpacing(45);
  //settings_layout->addLayout(sidebar_layout);
  //settings_layout->addSpacing(45);
  //settings_layout->addLayout(panel_layout);
  //settings_layout->addSpacing(45);
  //setLayout(settings_layout);

  setStyleSheet(R"(
    * {
      background-color: #000000;
      color: white;
      font-size: 50px;
      font-family: inter;
    }
  )");
}
