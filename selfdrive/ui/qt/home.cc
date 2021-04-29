#include <QDateTime>
#include <QHBoxLayout>
#include <QMouseEvent>
#include <QVBoxLayout>

#include "common/util.h"
#include "common/params.h"
#include "common/timing.h"
#include "common/swaglog.h"

#include "home.hpp"
#include "widgets/drive_stats.hpp"
#include "widgets/setup.hpp"

// HomeWindow: the container for the offroad and onroad UIs

HomeWindow::HomeWindow(QWidget* parent) : QWidget(parent) {
  layout = new QStackedLayout();
  layout->setStackingMode(QStackedLayout::StackAll);

  onroad = new OnroadWindow(this);
  layout->addWidget(onroad);
  QObject::connect(this, &HomeWindow::update, onroad, &OnroadWindow::update);
  QObject::connect(this, &HomeWindow::displayPowerChanged, onroad, &OnroadWindow::setEnabled);
  
  home = new OffroadHome();
  layout->addWidget(home);
  QObject::connect(this, &HomeWindow::openSettings, home, &OffroadHome::refresh);
  QObject::connect(this, &HomeWindow::offroadTransition, home, &OffroadHome::setVisible);

  setLayout(layout);
}

void HomeWindow::mousePressEvent(QMouseEvent* e) {
  // TODO: make a nice driver view widget
  if (QUIState::ui_state.scene.driver_view) {
    Params().putBool("IsDriverViewEnabled", false);
    QUIState::ui_state.scene.driver_view = false;
    return;
  }

  // Settings button click
  if (!QUIState::ui_state.sidebar_collapsed && settings_btn.ptInRect(e->x(), e->y())) {
    emit openSettings();
  }

  // Handle sidebar collapsing
  if (QUIState::ui_state.scene.started && (e->x() >= QUIState::ui_state.viz_rect.x - bdr_s)) {
    QUIState::ui_state.sidebar_collapsed = !QUIState::ui_state.sidebar_collapsed;
  }
}


// OffroadHome: the offroad home page

OffroadHome::OffroadHome(QWidget* parent) : QWidget(parent) {
  QVBoxLayout* main_layout = new QVBoxLayout();
  main_layout->setContentsMargins(sbr_w + 50, 50, 50, 50);

  // top header
  QHBoxLayout* header_layout = new QHBoxLayout();

  date = new QLabel();
  date->setStyleSheet(R"(font-size: 55px;)");
  header_layout->addWidget(date, 0, Qt::AlignHCenter | Qt::AlignLeft);

  alert_notification = new QPushButton();
  alert_notification->setVisible(false);
  QObject::connect(alert_notification, SIGNAL(released()), this, SLOT(openAlerts()));
  header_layout->addWidget(alert_notification, 0, Qt::AlignHCenter | Qt::AlignRight);

  std::string brand = Params().getBool("Passive") ? "dashcam" : "openpilot";
  QLabel* version = new QLabel(QString::fromStdString(brand + " v" + Params().get("Version")));
  version->setStyleSheet(R"(font-size: 55px;)");
  header_layout->addWidget(version, 0, Qt::AlignHCenter | Qt::AlignRight);

  main_layout->addLayout(header_layout);

  // main content
  main_layout->addSpacing(25);
  center_layout = new QStackedLayout();

  QHBoxLayout* statsAndSetup = new QHBoxLayout();
  statsAndSetup->setMargin(0);

  DriveStats* drive = new DriveStats;
  drive->setFixedSize(800, 800);
  statsAndSetup->addWidget(drive);

  SetupWidget* setup = new SetupWidget;
  //setup->setFixedSize(700, 700);
  statsAndSetup->addWidget(setup);

  QWidget* statsAndSetupWidget = new QWidget();
  statsAndSetupWidget->setLayout(statsAndSetup);

  center_layout->addWidget(statsAndSetupWidget);

  alerts_widget = new OffroadAlert();
  QObject::connect(alerts_widget, SIGNAL(closeAlerts()), this, SLOT(closeAlerts()));
  center_layout->addWidget(alerts_widget);
  center_layout->setAlignment(alerts_widget, Qt::AlignCenter);

  main_layout->addLayout(center_layout, 1);

  // set up refresh timer
  timer = new QTimer(this);
  QObject::connect(timer, SIGNAL(timeout()), this, SLOT(refresh()));
  refresh();
  timer->start(10 * 1000);

  setLayout(main_layout);
  setStyleSheet(R"(
    * {
     color: white;
    }
  )");
}

void OffroadHome::openAlerts() {
  center_layout->setCurrentIndex(1);
}

void OffroadHome::closeAlerts() {
  center_layout->setCurrentIndex(0);
}

void OffroadHome::refresh() {
  bool first_refresh = !date->text().size();
  if (!isVisible() && !first_refresh) {
    return;
  }

  date->setText(QDateTime::currentDateTime().toString("dddd, MMMM d"));

  // update alerts

  alerts_widget->refresh();
  if (!alerts_widget->alertCount && !alerts_widget->updateAvailable) {
    emit closeAlerts();
    alert_notification->setVisible(false);
    return;
  }

  if (alerts_widget->updateAvailable) {
    alert_notification->setText("UPDATE");
  } else {
    int alerts = alerts_widget->alertCount;
    alert_notification->setText(QString::number(alerts) + " ALERT" + (alerts == 1 ? "" : "S"));
  }

  if (!alert_notification->isVisible() && !first_refresh) {
    emit openAlerts();
  }
  alert_notification->setVisible(true);

  // Red background for alerts, blue for update available
  QString style = QString(R"(
    padding: 15px;
    padding-left: 30px;
    padding-right: 30px;
    border: 1px solid;
    border-radius: 5px;
    font-size: 40px;
    font-weight: 500;
    background-color: #E22C2C;
  )");
  if (alerts_widget->updateAvailable) {
    style.replace("#E22C2C", "#364DEF");
  }
  alert_notification->setStyleSheet(style);
}
