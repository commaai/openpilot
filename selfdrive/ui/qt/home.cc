#include "selfdrive/ui/qt/home.h"

#include <QDateTime>
#include <QHBoxLayout>
#include <QMouseEvent>
#include <QVBoxLayout>

#include "selfdrive/common/params.h"
#include "selfdrive/common/swaglog.h"
#include "selfdrive/common/timing.h"
#include "selfdrive/common/util.h"
#include "selfdrive/ui/qt/widgets/drive_stats.h"
#include "selfdrive/ui/qt/widgets/setup.h"
#include "selfdrive/ui/qt/util.h"

// HomeWindow: the container for the offroad and onroad UIs

HomeWindow::HomeWindow(QWidget* parent) : QWidget(parent) {
  QHBoxLayout *layout = new QHBoxLayout(this);
  layout->setMargin(0);
  layout->setSpacing(0);

  sidebar = new Sidebar(this);
  layout->addWidget(sidebar);
  QObject::connect(this, &HomeWindow::update, sidebar, &Sidebar::updateState);
  QObject::connect(sidebar, &Sidebar::openSettings, this, &HomeWindow::openSettings);

  slayout = new QStackedLayout();
  layout->addLayout(slayout);

  onroad = new OnroadWindow(this);
  slayout->addWidget(onroad);

  QObject::connect(this, &HomeWindow::update, onroad, &OnroadWindow::update);
  QObject::connect(this, &HomeWindow::offroadTransitionSignal, onroad, &OnroadWindow::offroadTransition);

  home = new OffroadHome();
  slayout->addWidget(home);
  QObject::connect(this, &HomeWindow::openSettings, home, &OffroadHome::refresh);

  setLayout(layout);
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

void HomeWindow::driverView() {
  if (!driver_view) {
    Params().putBool("IsDriverViewEnabled", true);
    driver_view = new DriverViewWindow(this);
    slayout->addWidget(driver_view);
    slayout->setCurrentWidget(driver_view);
    sidebar->setVisible(false);
    emit previewDriverCam();
  }
}

void HomeWindow::mousePressEvent(QMouseEvent* e) {
  // TODO: make a nice driver view widget
  if (driver_view) {
    Params().putBool("IsDriverViewEnabled", false);
    slayout->setCurrentWidget(home);
    driver_view->deleteLater();
    driver_view = nullptr;
    sidebar->setVisible(true);
    return;
  }

  // Handle sidebar collapsing
  if (onroad->isVisible() && (!sidebar->isVisible() || e->x() > sidebar->width())) {
    // Hide map first if visible, then hide sidebar
    if (onroad->map != nullptr && onroad->map->isVisible()){
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
  QVBoxLayout* main_layout = new QVBoxLayout();
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

  setLayout(main_layout);
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

// class DriverViewWindow

DriverViewWindow::DriverViewWindow(QWidget* parent) : sm({"driverState"}), QOpenGLWidget(parent) {
  setAttribute(Qt::WA_OpaquePaintEvent);
  is_rhd = Params().getBool("IsRHD");
  face_img = QImage("../assets/img_driver_face").scaled(180, 180, Qt::KeepAspectRatio, Qt::SmoothTransformation);
  timer = new QTimer(this);
  connect(timer, &QTimer::timeout, this, &DriverViewWindow::onTimeout);
}

DriverViewWindow::~DriverViewWindow() {
  makeCurrent();
  doneCurrent();
}

void DriverViewWindow::onTimeout() {
  vision->update();
  update();
}

void DriverViewWindow::initializeGL() {
  initializeOpenGLFunctions();
  vision = std::make_unique<UIVision>(VISION_STREAM_RGB_FRONT);
  timer->start(0);
}

void DriverViewWindow::paintGL() {
  const NVGcolor color = bg_colors[STATUS_DISENGAGED];
  glClearColor(color.r, color.g, color.b, 1.0);
  glClear(GL_STENCIL_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

  const Rect viz_rect = Rect{bdr_s, bdr_s, vwp_w - 2 * bdr_s, vwp_h - 2 * bdr_s};
  glViewport(viz_rect.x, viz_rect.y, viz_rect.w, viz_rect.h);
  QPainter p(this);

  if (!vision->connected()) {
    p.setPen(QColor(0xff, 0xff, 0xff));
    p.setRenderHint(QPainter::TextAntialiasing);
    configFont(p, "Open Sans", 100, "Bold");
    p.drawText(QRect{viz_rect.x, viz_rect.y, viz_rect.w, viz_rect.h}, Qt::AlignCenter, "Please wait for camera to start");
    return;
  }

  vision->draw();
  cereal::DriverState::Reader driver_state;
  sm.update(0);
  if (sm.updated("driverState")) {
    driver_state = sm["drive_state"].getDriverState();
  }

  const int width = 4 * viz_rect.h / 3;
  const Rect rect = {viz_rect.centerX() - width / 2, viz_rect.y, width, viz_rect.h};  // x, y, w, h
  const Rect valid_rect = {is_rhd ? rect.right() - rect.h / 2 : rect.x, rect.y, rect.h / 2, rect.h};

  // blackout
  const int blackout_x_r = valid_rect.right();
  const Rect& blackout_rect = Hardware::TICI() ? viz_rect : rect;
  const int blackout_w_r = blackout_rect.right() - valid_rect.right();
  const int blackout_x_l = blackout_rect.x;
  const int blackout_w_l = valid_rect.x - blackout_x_l;

  QColor bg;
  bg.setRgbF(0, 0, 0, 0.56);
  p.setBrush(QBrush(bg));
  p.drawRect(blackout_x_l, rect.y, blackout_w_l, rect.h);
  p.drawRect(blackout_x_r, rect.y, blackout_w_r, rect.h);
  p.setBrush(Qt::NoBrush);

  const bool face_detected = driver_state.getFaceProb() > 0.4;
  if (face_detected) {
    auto fxy_list = driver_state.getFacePosition();
    float face_x = fxy_list[0];
    float face_y = fxy_list[1];
    int fbox_x = valid_rect.centerX() + (is_rhd ? face_x : -face_x) * valid_rect.w;
    int fbox_y = valid_rect.centerY() + face_y * valid_rect.h;

    float alpha = 0.2;
    if (face_x = std::abs(face_x), face_y = std::abs(face_y); face_x <= 0.35 && face_y <= 0.4)
      alpha = 0.8 - (face_x > face_y ? face_x : face_y) * 0.6 / 0.375;

    const int box_size = 0.6 * rect.h / 2;
    QColor color;
    color.setRgbF(1.0, 1.0, 1.0, alpha);
    QPen pen = QPen(QColor(0xff, 0xff, 0xff, alpha));
    pen.setWidth(10);
    p.setPen(pen);
    p.drawRect(fbox_x - box_size / 2, fbox_y - box_size / 2, box_size, box_size);
    p.setPen(Qt::NoPen);
  }
  const int face_radius = 85;
  const int img_x = is_rhd ? rect.right() - face_radius * 2 - bdr_s * 2 : rect.x + bdr_s * 2;
  const int img_y = rect.bottom() - face_radius * 2 - bdr_s * 2.5;
  p.drawImage(img_x, img_y, face_img);
};
