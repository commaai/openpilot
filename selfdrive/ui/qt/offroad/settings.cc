#include <string>
#include <iostream>
#include <sstream>
#include <cassert>

#include <QString>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QPixmap>

#include "wifi.hpp"
#include "settings.hpp"
#include "widgets/toggle.hpp"
#include "widgets/offroad_alerts.hpp"

#include "common/params.h"
#include "common/util.h"


ParamsToggle::ParamsToggle(QString param, QString title, QString description, QString icon_path, QWidget *parent): QFrame(parent) , param(param) {
  QHBoxLayout *layout = new QHBoxLayout;
  layout->setSpacing(50);

  // Parameter image
  if (icon_path.length()) {
    QPixmap pix(icon_path);
    QLabel *icon = new QLabel();
    icon->setPixmap(pix.scaledToWidth(80, Qt::SmoothTransformation));
    icon->setSizePolicy(QSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed));
    layout->addWidget(icon);
  } else {
    layout->addSpacing(80);
  }

  // Name of the parameter
  QLabel *label = new QLabel(title);
  label->setStyleSheet(R"(font-size: 50px;)");
  layout->addWidget(label);

  // toggle switch
  Toggle *toggle = new Toggle(this);
  toggle->setFixedSize(150, 100);
  layout->addWidget(toggle);
  QObject::connect(toggle, SIGNAL(stateChanged(int)), this, SLOT(checkboxClicked(int)));

  // set initial state from param
  if (Params().read_db_bool(param.toStdString().c_str())) {
    toggle->togglePosition();
  }

  setLayout(layout);
}

void ParamsToggle::checkboxClicked(int state) {
  char value = state ? '1': '0';
  Params().write_db_value(param.toStdString().c_str(), &value, 1);
}

QWidget * toggles_panel() {
  QVBoxLayout *toggles_list = new QVBoxLayout();
  toggles_list->setMargin(50);
  toggles_list->setSpacing(25);

  toggles_list->addWidget(new ParamsToggle("OpenpilotEnabledToggle",
                                            "Enable openpilot",
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
  device_layout->setMargin(100);
  device_layout->setSpacing(50);

  Params params = Params();
  std::vector<std::pair<std::string, std::string>> labels = {
    {"Dongle ID", params.get("DongleId", false)},
  };

  // get serial number
  //std::string cmdline = util::read_file("/proc/cmdline");
  //auto delim = cmdline.find("serialno=");
  //if (delim != std::string::npos) {
  //  labels.push_back({"Serial", cmdline.substr(delim, cmdline.find(" ", delim))});
  //}

  for (auto &l : labels) {
    QString text = QString::fromStdString(l.first + ": " + l.second);
    device_layout->addWidget(new QLabel(text));
  }

  // TODO: show current calibration values
  QPushButton *clear_cal_btn = new QPushButton("Reset Calibration");
  device_layout->addWidget(clear_cal_btn);
  QObject::connect(clear_cal_btn, &QPushButton::released, [=]() {
    Params().delete_db_value("CalibrationParams");
  });

  QPushButton *poweroff_btn = new QPushButton("Power Off");
  device_layout->addWidget(poweroff_btn);
  QPushButton *reboot_btn = new QPushButton("Reboot");
  device_layout->addWidget(reboot_btn);
#ifdef __aarch64__
  QObject::connect(poweroff_btn, &QPushButton::released, [=]() { std::system("sudo poweroff"); });
  QObject::connect(reboot_btn, &QPushButton::released, [=]() { std::system("sudo reboot"); });
#endif

  // TODO: add confirmation dialog
  QPushButton *uninstall_btn = new QPushButton("Uninstall openpilot");
  device_layout->addWidget(uninstall_btn);
  QObject::connect(uninstall_btn, &QPushButton::released, [=]() { Params().write_db_value("DoUninstall", "1"); });

  QWidget *widget = new QWidget;
  widget->setLayout(device_layout);
  widget->setStyleSheet(R"(
    QPushButton {
      padding: 0;
      height: 120px;
      border-radius: 15px;
      background-color: #393939;
    }
  )");
  return widget;
}

QWidget * developer_panel() {
  QVBoxLayout *main_layout = new QVBoxLayout;
  main_layout->setMargin(100);

  // TODO: enable SSH toggle and github keys

  Params params = Params();
  std::string brand = params.read_db_bool("Passive") ? "dashcam" : "openpilot";
  std::vector<std::pair<std::string, std::string>> labels = {
    {"Version", brand + " v" + params.get("Version", false)},
    {"Git Branch", params.get("GitBranch", false)},
    {"Git Commit", params.get("GitCommit", false).substr(0, 10)},
    {"Panda Firmware", params.get("PandaFirmwareHex", false)},
  };

  std::string os_version = util::read_file("/VERSION");
  if (os_version.size()) {
    labels.push_back({"OS Version", "AGNOS " + os_version});
  }

  for (auto l : labels) {
    QString text = QString::fromStdString(l.first + ": " + l.second);
    main_layout->addWidget(new QLabel(text));
  }

  QWidget *widget = new QWidget;
  widget->setLayout(main_layout);
  widget->setStyleSheet(R"(
    QLabel {
      font-size: 50px;
    }
  )");
  return widget;
}

QWidget * network_panel(QWidget * parent) {
  WifiUI *w = new WifiUI();
  QObject::connect(w, SIGNAL(openKeyboard()), parent, SLOT(closeSidebar()));
  QObject::connect(w, SIGNAL(closeKeyboard()), parent, SLOT(openSidebar()));
  return w;
}


void SettingsWindow::setActivePanel() {
  auto *btn = qobject_cast<QPushButton *>(nav_btns->checkedButton());
  panel_layout->setCurrentWidget(panels[btn->text()]);
}

SettingsWindow::SettingsWindow(QWidget *parent) : QFrame(parent) {
  // setup two main layouts
  QVBoxLayout *sidebar_layout = new QVBoxLayout();
  sidebar_layout->setMargin(0);
  panel_layout = new QStackedLayout();

  // close button
  QPushButton *close_btn = new QPushButton("X");
  close_btn->setStyleSheet(R"(
    font-size: 90px;
    font-weight: bold;
    border 1px grey solid;
    border-radius: 100px;
    background-color: #292929;
  )");
  close_btn->setFixedSize(200, 200);
  sidebar_layout->addSpacing(45);
  sidebar_layout->addWidget(close_btn, 0, Qt::AlignLeft);
  QObject::connect(close_btn, SIGNAL(released()), this, SIGNAL(closeSettings()));

  // setup panels
  panels = {
    {"Developer", developer_panel()},
    {"Device", device_panel()},
    {"Network", network_panel(this)},
    {"Toggles", toggles_panel()},
  };

  sidebar_layout->addSpacing(45);
  nav_btns = new QButtonGroup();
  for (auto &panel : panels) {
    QPushButton *btn = new QPushButton(panel.first);
    btn->setCheckable(true);
    btn->setStyleSheet(R"(
      QPushButton {
        color: grey;
        border: none;
        background: none;
        font-size: 65px;
        font-weight: bold;
        padding-top: 35px;
        padding-bottom: 35px;
      }
      QPushButton:checked {
        color: white;
      }
    )");

    nav_btns->addButton(btn);
    sidebar_layout->addWidget(btn, 0, Qt::AlignRight | Qt::AlignTop);
    panel_layout->addWidget(panel.second);
    QObject::connect(btn, SIGNAL(released()), this, SLOT(setActivePanel()));
  }
  qobject_cast<QPushButton *>(nav_btns->buttons()[0])->setChecked(true);
  sidebar_layout->addStretch();

  // main settings layout, sidebar + main panel
  QHBoxLayout *settings_layout = new QHBoxLayout();
  settings_layout->setContentsMargins(150, 50, 150, 50);

  sidebar_widget = new QWidget;
  sidebar_widget->setLayout(sidebar_layout);
  settings_layout->addWidget(sidebar_widget);

  settings_layout->addSpacing(25);

  QFrame *panel_frame = new QFrame;
  panel_frame->setLayout(panel_layout);
  panel_frame->setStyleSheet(R"(
    QFrame {
      border-radius: 30px;
      background-color: #292929;
    }
    * {
      background-color: none;
    }
  )");
  settings_layout->addWidget(panel_frame, 1, Qt::AlignRight);

  setLayout(settings_layout);
  setStyleSheet(R"(
    * {
      color: white;
      font-size: 50px;
    }
    SettingsWindow {
      background-color: black;
    }
  )");
}

void SettingsWindow::closeSidebar() {
  sidebar_widget->setVisible(false);
}

void SettingsWindow::openSidebar() {
  sidebar_widget->setVisible(true);
}
