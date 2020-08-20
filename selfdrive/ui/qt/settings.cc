#include <string>
#include <iostream>
#include <sstream>
#include <cassert>

#include "qt/settings.hpp"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QScrollArea>
#include <QScroller>
#include <QScrollerProperties>
#include <QDebug>
#include <QPixmap>

#include "common/params.h"

ParamsToggle::ParamsToggle(QString param, QString title, QString description, QString icon, QWidget *parent): QFrame(parent) , param(param) {
  QHBoxLayout *hlayout = new QHBoxLayout;
  QVBoxLayout *vlayout = new QVBoxLayout;

  hlayout->addSpacing(25);
  if (icon.length()){
    QPixmap pix(icon);
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

  vlayout->addWidget(checkbox);
  vlayout->addWidget(label);
  hlayout->addLayout(vlayout);

  setLayout(hlayout);

  auto p = read_db_bytes(param.toStdString().c_str());
  if (p.size()){
    checkbox->setChecked(p[0] == '1');
  }

  setStyleSheet(R"(
    QCheckBox { font-size: 40px }
    QLabel { font-size: 20px }
    * {
      background-color: #114265;
    }
  )");

  QObject::connect(checkbox, SIGNAL(stateChanged(int)), this, SLOT(checkboxClicked(int)));
}

void ParamsToggle::checkboxClicked(int state){
  char value = state ? '1': '0';
  write_db_value(param.toStdString().c_str(), &value, 1);
}

SettingsWindow::SettingsWindow(QWidget *parent) : QWidget(parent) {
  QWidget *container = new QWidget(this);

  QVBoxLayout *settings_list = new QVBoxLayout();
  settings_list->addWidget(new ParamsToggle("OpenpilotEnabledToggle",
                                            "Enable Openpilot",
                                            "Use the openpilot system for adaptive cruise control and lane keep driver assistance. Your attention is required at all times to use this feature. Changing this setting takes effect when the car is powered off.",
                                            "../assets/offroad/icon_openpilot.png"
                                              ));
  settings_list->addWidget(new ParamsToggle("LaneChangeEnabled",
                                            "Enable Lane Change Assist",
                                            "Perform assisted lane changes with openpilot by checking your surroundings for safety, activating the turn signal and gently nudging the steering wheel towards your desired lane. openpilot is not capable of checking if a lane change is safe. You must continuously observe your surroundings to use this feature.",
                                            "../assets/offroad/icon_road.png"
                                              ));
  settings_list->addWidget(new ParamsToggle("IsLdwEnabled",
                                            "Enable Lane Departure Warnings",
                                            "Receive alerts to steer back into the lane when your vehicle drifts over a detected lane line without a turn signal activated while driving over 31mph (50kph).",
                                            "../assets/offroad/icon_warning.png"
                                              ));
  settings_list->addWidget(new ParamsToggle("RecordFront",
                                            "Record and Upload Driver Camera",
                                            "Upload data from the driver facing camera and help improve the driver monitoring algorithm.",
                                            "../assets/offroad/icon_network.png"
                                            ));
  settings_list->addWidget(new ParamsToggle("IsRHD",
                                            "Enable Right-Hand Drive",
                                            "Allow openpilot to obey left-hand traffic conventions and perform driver monitoring on right driver seat.",
                                            "../assets/offroad/icon_openpilot_mirrored.png"
                                            ));
  settings_list->addWidget(new ParamsToggle("IsMetric",
                                            "Use Metric System",
                                            "Display speed in km/h instead of mp/h.",
                                            "../assets/offroad/icon_metric.png"
                                            ));
  settings_list->addWidget(new ParamsToggle("CommunityFeaturesToggle",
                                            "Enable Community Features",
                                            "Use features from the open source community that are not maintained or supported by comma.ai and have not been confirmed to meet the standard safety model. These features include community supported cars and community supported hardware. Be extra cautious when using these features",
                                            "../assets/offroad/icon_shell.png"
                                            ));

  settings_list->setSpacing(25);

  container->setLayout(settings_list);
  container->setFixedWidth(1650);

  QScrollArea *scrollArea = new QScrollArea;
  scrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  scrollArea->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  scrollArea->setWidget(container);

  QScrollerProperties sp;
  sp.setScrollMetric(QScrollerProperties::DecelerationFactor, 2.0);

  QScroller* qs = QScroller::scroller(scrollArea);
  qs->setScrollerProperties(sp);

  QHBoxLayout *main_layout = new QHBoxLayout;
  main_layout->addSpacing(50);
  main_layout->addWidget(scrollArea);

  QPushButton * button = new QPushButton("Close");
  main_layout->addWidget(button);
  main_layout->addSpacing(20);

  setLayout(main_layout);

  QScroller::grabGesture(scrollArea, QScroller::LeftMouseButtonGesture);
  QObject::connect(button, SIGNAL(clicked()), this, SIGNAL(closeSettings()));

  setStyleSheet(R"(
    QPushButton { font-size: 40px }
  )");

}
