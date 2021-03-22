#include <string>
#include <iostream>
#include <sstream>
#include <cassert>

#ifndef QCOM
#include "networking.hpp"
#endif
#include "settings.hpp"
#include "widgets/input.hpp"
#include "widgets/toggle.hpp"
#include "widgets/offroad_alerts.hpp"
#include "widgets/controls.hpp"
#include "common/params.h"
#include "common/util.h"
#include "selfdrive/hardware/hw.h"


QWidget * toggles_panel() {
  QVBoxLayout *toggles_list = new QVBoxLayout();

  toggles_list->setMargin(50);
  toggles_list->addWidget(new ToggleControl("OpenpilotEnabledToggle",
                                            "Enable openpilot",
                                            "Use the openpilot system for adaptive cruise control and lane keep driver assistance. Your attention is required at all times to use this feature. Changing this setting takes effect when the car is powered off.",
                                            "../assets/offroad/icon_openpilot.png"
                                              ));
  toggles_list->addWidget(horizontal_line());
  toggles_list->addWidget(new ToggleControl("LaneChangeEnabled",
                                            "Enable Lane Change Assist",
                                            "Perform assisted lane changes with openpilot by checking your surroundings for safety, activating the turn signal and gently nudging the steering wheel towards your desired lane. openpilot is not capable of checking if a lane change is safe. You must continuously observe your surroundings to use this feature.",
                                            "../assets/offroad/icon_road.png"
                                              ));
  toggles_list->addWidget(horizontal_line());
  toggles_list->addWidget(new ToggleControl("IsLdwEnabled",
                                            "Enable Lane Departure Warnings",
                                            "Receive alerts to steer back into the lane when your vehicle drifts over a detected lane line without a turn signal activated while driving over 31mph (50kph).",
                                            "../assets/offroad/icon_warning.png"
                                              ));
  toggles_list->addWidget(horizontal_line());
  toggles_list->addWidget(new ToggleControl("IsRHD",
                                            "Enable Right-Hand Drive",
                                            "Allow openpilot to obey left-hand traffic conventions and perform driver monitoring on right driver seat.",
                                            "../assets/offroad/icon_openpilot_mirrored.png"
                                            ));
  toggles_list->addWidget(horizontal_line());
  toggles_list->addWidget(new ToggleControl("IsMetric",
                                            "Use Metric System",
                                            "Display speed in km/h instead of mp/h.",
                                            "../assets/offroad/icon_metric.png"
                                            ));
  toggles_list->addWidget(horizontal_line());
  toggles_list->addWidget(new ToggleControl("CommunityFeaturesToggle",
                                            "Enable Community Features",
                                            "Use features from the open source community that are not maintained or supported by comma.ai and have not been confirmed to meet the standard safety model. These features include community supported cars and community supported hardware. Be extra cautious when using these features",
                                            "../assets/offroad/icon_shell.png"
                                            ));

  ToggleControl *record_toggle = new ToggleControl("RecordFront",
                                            "Record and Upload Driver Camera",
                                            "Upload data from the driver facing camera and help improve the driver monitoring algorithm.",
                                            "../assets/offroad/icon_network.png");
  toggles_list->addWidget(horizontal_line());
  toggles_list->addWidget(record_toggle);

  bool record_lock = Params().read_db_bool("RecordFrontLock");
  record_toggle->setEnabled(!record_lock);

  QWidget *widget = new QWidget;
  widget->setLayout(toggles_list);
  return widget;
}

QWidget * device_panel() {
  QVBoxLayout *device_layout = new QVBoxLayout;
  device_layout->setMargin(100);

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
    device_layout->addWidget(new LabelControl(QString::fromStdString(l.first),
                             QString::fromStdString(l.second)));
  }

  device_layout->addWidget(horizontal_line());

  device_layout->addWidget(new ButtonControl("Driver Camera", "PREVIEW",
                                             "Preview the driver facing camera to help optimize device mounting position for best driver monitoring experience. (vehicle must be off)",
                                             [=]() { Params().write_db_value("IsDriverViewEnabled", "1", 1); }));

  device_layout->addWidget(horizontal_line());

  // TODO: show current calibration values
  device_layout->addWidget(new ButtonControl("Reset Calibration", "RESET",
                                             "openpilot requires the device to be mounted within 4° left or right and within 5° up or down. openpilot is continuously calibrating, resetting is rarely required.",
                                             [=]() {
                                               if (ConfirmationDialog::confirm("Are you sure you want to reset calibration?")) {
                                                 Params().delete_db_value("CalibrationParams");
                                               }
                                             }));

  // power buttons

  QPushButton *poweroff_btn = new QPushButton("Power Off");
  device_layout->addWidget(poweroff_btn, Qt::AlignBottom);
  QObject::connect(poweroff_btn, &QPushButton::released, [=]() {
    if (ConfirmationDialog::confirm("Are you sure you want to power off?")) {
      Hardware::poweroff();
    }
  });

  device_layout->addWidget(horizontal_line(), Qt::AlignBottom);

  QPushButton *reboot_btn = new QPushButton("Reboot");
  device_layout->addWidget(reboot_btn, Qt::AlignBottom);
  QObject::connect(reboot_btn, &QPushButton::released, [=]() {
    if (ConfirmationDialog::confirm("Are you sure you want to reboot?")) {
      Hardware::reboot();
    }
  });

  QPushButton *uninstall_btn = new QPushButton("Uninstall openpilot");
  device_layout->addWidget(uninstall_btn);
  QObject::connect(uninstall_btn, &QPushButton::released, [=]() {
    if (ConfirmationDialog::confirm("Are you sure you want to uninstall?")) {
      Params().write_db_value("DoUninstall", "1");
    }
  });

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

DeveloperPanel::DeveloperPanel(QWidget* parent) : QFrame(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setMargin(100);
  setLayout(main_layout);
  setStyleSheet(R"(QLabel {font-size: 50px;})");
}

void DeveloperPanel::showEvent(QShowEvent *event) {
  Params params = Params();
  std::string brand = params.read_db_bool("Passive") ? "dashcam" : "openpilot";
  QList<QPair<QString, std::string>> dev_params = {
    {"Version", brand + " v" + params.get("Version", false)},
    {"Git Branch", params.get("GitBranch", false)},
    {"Git Commit", params.get("GitCommit", false).substr(0, 10)},
    {"Panda Firmware", params.get("PandaFirmwareHex", false)},
    {"OS Version", Hardware::get_os_version()},
  };

  for (int i = 0; i < dev_params.size(); i++) {
    const auto &[name, value] = dev_params[i];
    if (labels.size() > i) {
      labels[i]->setText(QString::fromStdString(value));
    } else {
      labels.push_back(new LabelControl(name, QString::fromStdString(value)));
      layout()->addWidget(labels[i]);
      if (i < (dev_params.size() - 1)) {
        layout()->addWidget(horizontal_line());
      }
    }
  }
}

QWidget * network_panel(QWidget * parent) {
#ifdef QCOM
  QVBoxLayout *layout = new QVBoxLayout;
  layout->setMargin(100);
  layout->setSpacing(30);

  // TODO: can probably use the ndk for this
  // simple wifi + tethering buttons
  std::vector<std::pair<const char*, const char*>> btns = {
    {"WiFi Settings", "am start -n com.android.settings/.wifi.WifiPickerActivity \
                       -a android.net.wifi.PICK_WIFI_NETWORK \
                       --ez extra_prefs_show_button_bar true \
                       --es extra_prefs_set_next_text ''"},
    {"Tethering Settings", "am start -n com.android.settings/.TetherSettings \
                            --ez extra_prefs_show_button_bar true \
                            --es extra_prefs_set_next_text ''"},
  };
  for (auto &b : btns) {
    layout->addWidget(new ButtonControl(b.first, "OPEN", "", [=]() { std::system(b.second); }));
  }
  layout->addStretch(1);

  QWidget *w = new QWidget;
  w->setLayout(layout);
#else
  Networking *w = new Networking(parent);
#endif
  return w;
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
  sidebar_layout->addWidget(close_btn, 0, Qt::AlignCenter);
  QObject::connect(close_btn, SIGNAL(released()), this, SIGNAL(closeSettings()));

  // setup panels
  QPair<QString, QWidget *> panels[] = {
    {"Device", device_panel()},
    {"Network", network_panel(this)},
    {"Toggles", toggles_panel()},
    {"Developer", new DeveloperPanel()},
  };

  sidebar_layout->addSpacing(45);
  nav_btns = new QButtonGroup();
  for (auto &[name, panel] : panels) {
    QPushButton *btn = new QPushButton(name);
    btn->setCheckable(true);
    btn->setStyleSheet(R"(
      * {
        color: grey;
        border: none;
        background: none;
        font-size: 65px;
        font-weight: 500;
        padding-top: 35px;
        padding-bottom: 35px;
      }
      QPushButton:checked {
        color: white;
      }
    )");

    nav_btns->addButton(btn);
    sidebar_layout->addWidget(btn, 0, Qt::AlignRight);

    panel_layout->addWidget(panel);
    QObject::connect(btn, &QPushButton::released, [=, w = panel]() { panel_layout->setCurrentWidget(w); });
  }
  qobject_cast<QPushButton *>(nav_btns->buttons()[0])->setChecked(true);
  sidebar_layout->setContentsMargins(50, 50, 100, 50);

  // main settings layout, sidebar + main panel
  QHBoxLayout *settings_layout = new QHBoxLayout();

  sidebar_widget = new QWidget;
  sidebar_widget->setLayout(sidebar_layout);
  sidebar_widget->setFixedWidth(500);
  settings_layout->addWidget(sidebar_widget);


  panel_frame = new QFrame;
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
  settings_layout->addWidget(panel_frame);

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
