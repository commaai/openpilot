#include "selfdrive/ui/qt/home.h"

#include <QDateTime>
#include <QHBoxLayout>
#include <QMouseEvent>
#include <QVBoxLayout>

#include "selfdrive/common/params.h"
#include "selfdrive/common/swaglog.h"
#include "selfdrive/common/timing.h"
#include "selfdrive/common/util.h"
#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/qt/widgets/drive_stats.h"
#include "selfdrive/ui/qt/widgets/setup.h"

// HomeWindow: the container for the offroad and onroad UIs

HomeWindow::HomeWindow(QWidget* parent) : QWidget(parent) {
  QHBoxLayout *main_layout = new QHBoxLayout(this);
  main_layout->setMargin(0);
  main_layout->setSpacing(0);

  sidebar = new Sidebar(this);
  main_layout->addWidget(sidebar);
  QObject::connect(this, &HomeWindow::update, sidebar, &Sidebar::updateState);
  QObject::connect(sidebar, &Sidebar::openSettings, this, &HomeWindow::openSettings);

  slayout = new QStackedLayout();
  main_layout->addLayout(slayout);

  onroad = new OnroadWindow(this);
  slayout->addWidget(onroad);

  QObject::connect(this, &HomeWindow::update, onroad, &OnroadWindow::update);
  QObject::connect(this, &HomeWindow::offroadTransitionSignal, onroad, &OnroadWindow::offroadTransitionSignal);

  home = new OffroadHome();
  slayout->addWidget(home);
  QObject::connect(this, &HomeWindow::openSettings, home, &OffroadHome::refresh);

  driver_view = new DriverViewWindow(this);
  connect(driver_view, &DriverViewWindow::done, [=] {
    showDriverView(false);
  });
  slayout->addWidget(driver_view);
}

void HomeWindow::offroadTransition(bool offroad) {
  if (offroad) {
    slayout->setCurrentWidget(home);
  } else {
    slayout->setCurrentWidget(onroad);
  }
  sidebar->setVisible(offroad);
  emit offroadTransitionSignal(offroad);
}

void HomeWindow::showDriverView(bool show) {
  if (show) {
    emit closeSettings();
    slayout->setCurrentWidget(driver_view);
  } else {
    slayout->setCurrentWidget(home);
  }
  sidebar->setVisible(show == false);
}

void HomeWindow::mousePressEvent(QMouseEvent* e) {
  // Handle sidebar collapsing
  if (onroad->isVisible() && (!sidebar->isVisible() || e->x() > sidebar->width())) {

    // TODO: Handle this without exposing pointer to map widget
    // Hide map first if visible, then hide sidebar
    if (onroad->map != nullptr && onroad->map->isVisible()) {
      onroad->map->setVisible(false);
    } else if (!sidebar->isVisible()) {
      sidebar->setVisible(true);
    } else {
      sidebar->setVisible(false);

      if (onroad->map != nullptr) onroad->map->setVisible(true);
    }
  }
}

// OffroadHome: the offroad home page

OffroadHome::OffroadHome(QWidget* parent) : QFrame(parent) {
  QVBoxLayout* main_layout = new QVBoxLayout(this);
  main_layout->setMargin(50);

  // top header
  QHBoxLayout* header_layout = new QHBoxLayout();

  date = new QLabel();
  date->setStyleSheet(R"(font-size: 55px;)");
  header_layout->addWidget(date, 0, Qt::AlignHCenter | Qt::AlignLeft);

  alert_notification = new QPushButton();
  alert_notification->setVisible(false);
  QObject::connect(alert_notification, &QPushButton::released, this, &OffroadHome::openAlerts);
  header_layout->addWidget(alert_notification, 0, Qt::AlignHCenter | Qt::AlignRight);

  QLabel* version = new QLabel(getBrandVersion());
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
  statsAndSetup->addWidget(setup);

  QWidget* statsAndSetupWidget = new QWidget();
  statsAndSetupWidget->setLayout(statsAndSetup);

  center_layout->addWidget(statsAndSetupWidget);

  alerts_widget = new OffroadAlert();
  QObject::connect(alerts_widget, &OffroadAlert::closeAlerts, this, &OffroadHome::closeAlerts);
  center_layout->addWidget(alerts_widget);
  center_layout->setAlignment(alerts_widget, Qt::AlignCenter);

  main_layout->addLayout(center_layout, 1);

  // set up refresh timer
  timer = new QTimer(this);
  QObject::connect(timer, &QTimer::timeout, this, &OffroadHome::refresh);
  timer->start(10 * 1000);

  setStyleSheet(R"(
    OffroadHome {
      background-color: black;
    }
    * {
     color: white;
    }
  )");
}

void OffroadHome::showEvent(QShowEvent *event) {
  refresh();
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
    closeAlerts();
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
    openAlerts();
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
