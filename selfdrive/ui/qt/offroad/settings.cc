#include <string>
#include <iostream>
#include <sstream>
#include <cassert>

#include "settings.hpp"

#include <QString>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QPixmap>

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

SettingsWindow::SettingsWindow(QWidget *parent) : QWidget(parent) {

  // sidebar
  QVBoxLayout *sidebar_layout = new QVBoxLayout();
  panel_layout = new QStackedLayout();

  // close button
  QPushButton *close_button = new QPushButton("X");
  close_button->setStyleSheet(R"(
    QPushButton {
      padding: 50px;
      font-weight: bold;
      font-size: 100px;
    }
  )");
  sidebar_layout->addWidget(close_button);
  QObject::connect(close_button, SIGNAL(released()), this, SIGNAL(closeSettings()));

  // setup panels
  panels = {
    {"device", device_panel()},
    {"toggles", toggles_panel()},
    {"developer", developer_panel()},
  };

  for (auto &panel : panels) {
    QPushButton *btn = new QPushButton(panel.first);
    btn->setStyleSheet(R"(
      QPushButton {
        padding-top: 35px;
        padding-bottom: 35px;
        font-size: 60px;
        text-align: right;
        border: none;
        background: none;
        font-weight: bold;
      }
    )");

    sidebar_layout->addWidget(btn);
    panel_layout->addWidget(panel.second);
    QObject::connect(btn, SIGNAL(released()), this, SLOT(setActivePanel()));
  }

  QHBoxLayout *settings_layout = new QHBoxLayout();
  settings_layout->addSpacing(45);
  settings_layout->addLayout(sidebar_layout);
  settings_layout->addSpacing(45);
  settings_layout->addLayout(panel_layout);
  settings_layout->addSpacing(45);
  setLayout(settings_layout);

  setStyleSheet(R"(
    * {
      color: white;
      font-size: 50px;
    }
  )");
}
