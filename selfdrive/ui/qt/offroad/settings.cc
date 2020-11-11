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

SettingsWindow::SettingsWindow(QWidget *parent) : QWidget(parent) {


  QWidget *lWidget = new QWidget();
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
  {
    QLabel *settingsLabel = new ClickableLabel();
    settingsLabel->setText("General");
    left_panel_layout->addWidget(settingsLabel);
  }
  {
    QLabel *settingsLabel = new ClickableLabel();
    settingsLabel->setText("Device");
    left_panel_layout->addWidget(settingsLabel);
  }
  {
    QLabel *settingsLabel = new ClickableLabel();
    settingsLabel->setText("Network");
    left_panel_layout->addWidget(settingsLabel);
  }
  {
    QLabel *settingsLabel = new ClickableLabel();
    settingsLabel->setText("Developer");
    left_panel_layout->addWidget(settingsLabel);
  }

  left_panel_layout->addStretch(3);
  lWidget->setLayout(left_panel_layout);

  QWidget *rWidget = new QWidget();
  rWidget->setStyleSheet(R"(
    * {
      background-color: #292929;
      font-size: 30px;
      padding-left: 30px;
      padding-right: 30px;
      padding-top: 10px;
    }
  )");

  QSizePolicy rightPolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
  rightPolicy.setHorizontalStretch(3);
  rWidget->setSizePolicy(rightPolicy);

  QVBoxLayout *general_settings_layout = new QVBoxLayout();

  {
    QHBoxLayout *row_layout = new QHBoxLayout();
    auto text = new QLabel("General");
    text->setStyleSheet(R"(
      font-size: 50px;
      font-weight: bold;
    )");
    QSizePolicy text_policy(QSizePolicy::Preferred, QSizePolicy::Preferred);
    text_policy.setHorizontalStretch(8);
    text->setSizePolicy(text_policy);

    auto btn = new QPushButton("?");
    QSizePolicy btn_policy(QSizePolicy::Preferred, QSizePolicy::Preferred);
    btn_policy.setHorizontalStretch(1);
    btn->setSizePolicy(btn_policy);

    row_layout->addWidget(text);
    row_layout->addWidget(btn);
    general_settings_layout->addStretch(1);
    general_settings_layout->addLayout(row_layout);
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

    // Sigh... line break won't obey the padding so we need to give it a margin.
    QWidget *line_break = new QWidget();
    line_break->setFixedHeight(2);
    line_break->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    line_break->setStyleSheet(R"(
        background-color: #414141;
        margin-left: 30px;
      )");
    general_settings_layout->addStretch(1);
    general_settings_layout->addWidget(line_break);
    general_settings_layout->addStretch(1);
  }

  general_settings_layout->addStretch(general_text_items.size());
  rWidget->setLayout(general_settings_layout);

  QHBoxLayout *panel_separator_layout = new QHBoxLayout();
  panel_separator_layout->addWidget(lWidget);
  panel_separator_layout->addWidget(rWidget);

  setLayout(panel_separator_layout);
  //panel_layout = new QStackedLayout();

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
