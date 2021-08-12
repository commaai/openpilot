#include "selfdrive/ui/qt/onroad.h"

#include <iostream>
#include <QDebug>

#include "selfdrive/common/swaglog.h"
#include "selfdrive/common/timing.h"
#include "selfdrive/ui/paint.h"
#include "selfdrive/ui/qt/util.h"
#ifdef ENABLE_MAPS
#include "selfdrive/ui/qt/maps/map.h"
#endif

OnroadWindow::OnroadWindow(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout  = new QVBoxLayout(this);
  main_layout->setMargin(bdr_s);
  QStackedLayout *stacked_layout = new QStackedLayout;
  stacked_layout->setStackingMode(QStackedLayout::StackAll);
  main_layout->addLayout(stacked_layout);

  // old UI on bottom
  nvg = new NvgWindow(this);
  QObject::connect(nvg, &NvgWindow::resizeSignal, [=](int w, int h){
    buttons->setFixedWidth(w);
  });
  QObject::connect(this, &OnroadWindow::updateStateSignal, nvg, &NvgWindow::updateState);

  QWidget * split_wrapper = new QWidget;
  split = new QHBoxLayout(split_wrapper);
  split->setContentsMargins(0, 0, 0, 0);
  split->setSpacing(0);
  split->addWidget(nvg);

  buttons = new ButtonsWindow(this);
  QObject::connect(this, &OnroadWindow::updateStateSignal, buttons, &ButtonsWindow::updateState);
  stacked_layout->addWidget(buttons);

  stacked_layout->addWidget(split_wrapper);

  alerts = new OnroadAlerts(this);
  alerts->setAttribute(Qt::WA_TransparentForMouseEvents, true);
  stacked_layout->addWidget(alerts);

  // setup stacking order
  alerts->raise();

  setAttribute(Qt::WA_OpaquePaintEvent);
  QObject::connect(this, &OnroadWindow::updateStateSignal, this, &OnroadWindow::updateState);
  QObject::connect(this, &OnroadWindow::offroadTransitionSignal, this, &OnroadWindow::offroadTransition);
}

void OnroadWindow::updateState(const UIState &s) {
  SubMaster &sm = *(s.sm);
  QColor bgColor = bg_colors[s.status];
  if (sm.updated("controlsState")) {
    const cereal::ControlsState::Reader &cs = sm["controlsState"].getControlsState();
    alerts->updateAlert({QString::fromStdString(cs.getAlertText1()),
                 QString::fromStdString(cs.getAlertText2()),
                 QString::fromStdString(cs.getAlertType()),
                 cs.getAlertSize(), cs.getAlertSound()}, bgColor);
  } else if ((sm.frame - s.scene.started_frame) > 5 * UI_FREQ) {
    // Handle controls timeout
    if (sm.rcv_frame("controlsState") < s.scene.started_frame) {
      // car is started, but controlsState hasn't been seen at all
      alerts->updateAlert(CONTROLS_WAITING_ALERT, bgColor);
    } else if ((nanos_since_boot() - sm.rcv_time("controlsState")) / 1e9 > CONTROLS_TIMEOUT) {
      // car is started, but controls is lagging or died
      bgColor = bg_colors[STATUS_ALERT];
      alerts->updateAlert(CONTROLS_UNRESPONSIVE_ALERT, bgColor);
    }
  }
  if (bg != bgColor) {
    // repaint border
    bg = bgColor;
    update();
  }
}

void OnroadWindow::offroadTransition(bool offroad) {
#ifdef ENABLE_MAPS
  if (!offroad) {
    QString token = QString::fromStdString(Params().get("MapboxToken"));
    if (map == nullptr && !token.isEmpty()) {
      QMapboxGLSettings settings;
      if (!Hardware::PC()) {
        settings.setCacheDatabasePath("/data/mbgl-cache.db");
      }
      settings.setCacheDatabaseMaximumSize(20 * 1024 * 1024);
      settings.setAccessToken(token.trimmed());

      MapWindow * m = new MapWindow(settings);
      m->setFixedWidth(width() / 2 - bdr_s);
      QObject::connect(this, &OnroadWindow::offroadTransitionSignal, m, &MapWindow::offroadTransition);
      split->addWidget(m, 0, Qt::AlignRight);
      map = m;
    }
  }
#endif

  alerts->updateAlert({}, bg);
}

void OnroadWindow::paintEvent(QPaintEvent *event) {
  QPainter p(this);
  p.fillRect(rect(), QColor(bg.red(), bg.green(), bg.blue(), 255));
}

// ***** onroad widgets *****

ButtonsWindow::ButtonsWindow(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout  = new QVBoxLayout(this);

  QWidget *btns_wrapper = new QWidget;
  QHBoxLayout *btns_layout  = new QHBoxLayout(btns_wrapper);
  btns_layout->setSpacing(0);
  btns_layout->setContentsMargins(0, 0, 30, 30);

  main_layout->addWidget(btns_wrapper, 0, Qt::AlignBottom);

//  mlButton = new QPushButton("Toggle Model Long");
//  mlButton->setStyleSheet("font-size: 50px; border-radius: 25px; border-color: #b83737;");
//  QObject::connect(mlButton, &QPushButton::clicked, [=]() {
//    mlButton->setStyleSheet("font-size: 50px; border-radius: 25px; border-color: #37b868;");
//  });
//  mlButton->setFixedWidth(525);
//  mlButton->setFixedHeight(150);
//  btns_layout->addStretch();
//  btns_layout->addWidget(mlButton, 0, Qt::AlignCenter);

  dfButton = new QPushButton("DF\nprofile");
  QObject::connect(dfButton, &QPushButton::clicked, [=]() {
    QUIState::ui_state.scene.dfButtonStatus = dfStatus < 3 ? dfStatus + 1 : 0;  // wrap back around
  });
  dfButton->setFixedWidth(200);
  dfButton->setFixedHeight(200);
//  btns_layout->addStretch();
  btns_layout->addWidget(dfButton, 0, Qt::AlignRight);

  setStyleSheet(R"(
    QPushButton {
      color: white;
      text-align: center;
      padding: 0px;
      border-width: 12px;
      border-style: solid;
      background-color: rgba(75, 75, 75, 0.3);
    }
  )");
}

void ButtonsWindow::updateState(const UIState &s) {
  updateDfButton(s.scene.dfButtonStatus);  // update dynamic follow profile button
//  updateMlButton(s.scene.dfButtonStatus);  // update model longitudinal button  // TODO: add model long back first
}

void ButtonsWindow::updateDfButton(int status) {
  if (dfStatus != status) {  // updates (resets) on car start, or on button press
    dfStatus = status;
    dfButton->setStyleSheet(QString("font-size: 45px; border-radius: 100px; border-color: %1").arg(dfButtonColors.at(dfStatus)));

    MessageBuilder msg;
    auto dfButtonStatus = msg.initEvent().initDynamicFollowButton();
    dfButtonStatus.setStatus(dfStatus);
    QUIState::ui_state.pm->send("dynamicFollowButton", msg);
  }
}

void OnroadAlerts::updateAlert(const Alert &a, const QColor &color) {
  if (!alert.equal(a) || color != bg) {
    alert = a;
    bg = color;
    update();
  }
}

void OnroadAlerts::paintEvent(QPaintEvent *event) {
  if (alert.size == cereal::ControlsState::AlertSize::NONE) {
    return;
  }
  static std::map<cereal::ControlsState::AlertSize, const int> alert_sizes = {
    {cereal::ControlsState::AlertSize::SMALL, 271},
    {cereal::ControlsState::AlertSize::MID, 420},
    {cereal::ControlsState::AlertSize::FULL, height()},
  };
  int h = alert_sizes[alert.size];
  QRect r = QRect(0, height() - h, width(), h);

  QPainter p(this);

  // draw background + gradient
  p.setPen(Qt::NoPen);
  p.setCompositionMode(QPainter::CompositionMode_SourceOver);

  p.setBrush(QBrush(bg));
  p.drawRect(r);

  QLinearGradient g(0, r.y(), 0, r.bottom());
  g.setColorAt(0, QColor::fromRgbF(0, 0, 0, 0.05));
  g.setColorAt(1, QColor::fromRgbF(0, 0, 0, 0.35));

  p.setCompositionMode(QPainter::CompositionMode_DestinationOver);
  p.setBrush(QBrush(g));
  p.fillRect(r, g);
  p.setCompositionMode(QPainter::CompositionMode_SourceOver);

  // text
  const QPoint c = r.center();
  p.setPen(QColor(0xff, 0xff, 0xff));
  p.setRenderHint(QPainter::TextAntialiasing);
  if (alert.size == cereal::ControlsState::AlertSize::SMALL) {
    configFont(p, "Open Sans", 74, "SemiBold");
    p.drawText(r, Qt::AlignCenter, alert.text1);
  } else if (alert.size == cereal::ControlsState::AlertSize::MID) {
    configFont(p, "Open Sans", 88, "Bold");
    p.drawText(QRect(0, c.y() - 125, width(), 150), Qt::AlignHCenter | Qt::AlignTop, alert.text1);
    configFont(p, "Open Sans", 66, "Regular");
    p.drawText(QRect(0, c.y() + 21, width(), 90), Qt::AlignHCenter, alert.text2);
  } else if (alert.size == cereal::ControlsState::AlertSize::FULL) {
    bool l = alert.text1.length() > 15;
    configFont(p, "Open Sans", l ? 132 : 177, "Bold");
    p.drawText(QRect(0, r.y() + (l ? 240 : 270), width(), 600), Qt::AlignHCenter | Qt::TextWordWrap, alert.text1);
    configFont(p, "Open Sans", 88, "Regular");
    p.drawText(QRect(0, r.height() - (l ? 361 : 420), width(), 300), Qt::AlignHCenter | Qt::TextWordWrap, alert.text2);
  }
}


NvgWindow::NvgWindow(QWidget *parent) : QOpenGLWidget(parent) {
  setAttribute(Qt::WA_OpaquePaintEvent);
}

NvgWindow::~NvgWindow() {
  makeCurrent();
  doneCurrent();
}

void NvgWindow::initializeGL() {
  initializeOpenGLFunctions();
  qInfo() << "OpenGL version:" << QString((const char*)glGetString(GL_VERSION));
  qInfo() << "OpenGL vendor:" << QString((const char*)glGetString(GL_VENDOR));
  qInfo() << "OpenGL renderer:" << QString((const char*)glGetString(GL_RENDERER));
  qInfo() << "OpenGL language version:" << QString((const char*)glGetString(GL_SHADING_LANGUAGE_VERSION));

  ui_nvg_init(&QUIState::ui_state);
  prev_draw_t = millis_since_boot();
}

void NvgWindow::updateState(const UIState &s) {
  // Connecting to visionIPC requires opengl to be current
  if (s.vipc_client->connected) {
    makeCurrent();
  }
  if (isVisible() != s.vipc_client->connected) {
    setVisible(s.vipc_client->connected);
  }
  repaint();
}

void NvgWindow::resizeGL(int w, int h) {
  ui_resize(&QUIState::ui_state, w, h);
  emit resizeSignal(w, h);  // for ButtonsWindow
}

void NvgWindow::paintGL() {
  ui_draw(&QUIState::ui_state, width(), height());

  double cur_draw_t = millis_since_boot();
  double dt = cur_draw_t - prev_draw_t;
  if (dt > 66) {
    // warn on sub 15fps
    LOGW("slow frame time: %.2f", dt);
  }
  prev_draw_t = cur_draw_t;
}
