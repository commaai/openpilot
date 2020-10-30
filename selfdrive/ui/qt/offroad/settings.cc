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

  vlayout->addSpacing(50);
  vlayout->addWidget(checkbox);
  vlayout->addWidget(label);
  vlayout->addSpacing(50);
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

  std::vector<std::pair<QString, QString>> labels = {
    {"Serial Number", "abcdefghijk"},
    {"Dongle ID", "abcdefghijk"},
  };

  for (auto l : labels) {
    device_layout->addWidget(new QLabel(l.first));
  }

  QWidget *widget = new QWidget;
  widget->setLayout(device_layout);
  return widget;
}

QWidget * developer_panel() {
  QVBoxLayout *developer_layout = new QVBoxLayout;

  Params params = Params();
  std::vector<std::pair<QString, QString>> labels = {
    {"Version", QString::fromStdString(params.get("Version", false))},
    {"Git Branch", QString::fromStdString(params.get("GitBranch", false))},
    {"Git Commit", QString::fromStdString(params.get("GitCommit", false))},
  };

  for (auto l : labels) {
    developer_layout->addWidget(new QLabel(l.first + QString(": ") + l.second));
  }

  QWidget *widget = new QWidget;
  widget->setLayout(developer_layout);
  return widget;
}



void SettingsWindow::setActivePanel() {
  QPushButton* btn = qobject_cast<QPushButton*>(sender());
  panel_layout->setCurrentWidget(panels[btn->text()]);
}

SettingsWindow::SettingsWindow(QWidget *parent) : QWidget(parent) {

  panels = {
    {"device", device_panel()},
    {"developer", developer_panel()},
    {"toggles", toggles_panel()},
  };

  // sidebar
  QVBoxLayout *sidebar_layout = new QVBoxLayout();
  panel_layout = new QStackedLayout();

  // close button
  QPushButton * close_button = new QPushButton("<- back");
  sidebar_layout->addWidget(close_button);
  QObject::connect(close_button, SIGNAL(released()), this, SIGNAL(closeSettings()));

  for (auto &panel : panels) {
    QPushButton * btn = new QPushButton(panel.first);
    sidebar_layout->addWidget(btn);
    panel_layout->addWidget(panel.second);
    QObject::connect(btn, SIGNAL(released()), this, SLOT(setActivePanel()));
  }

  QHBoxLayout *settings_layout = new QHBoxLayout();
  settings_layout->addLayout(sidebar_layout);
  settings_layout->addSpacing(10);
  settings_layout->addLayout(panel_layout);
  setLayout(settings_layout);

  setStyleSheet(R"(
    * {
      color: white;
      background-color: #072339;
      font-size: 60px;
    }
    QPushButton {
      font-size: 60px;
    }
  )");
}
