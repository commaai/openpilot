#include "selfdrive/ui/qt/onroad.h"

#include <QDebug>

#include "selfdrive/common/timing.h"
#include "selfdrive/ui/paint.h"
#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/qt/api.h"
#ifdef ENABLE_MAPS
#include "selfdrive/ui/qt/maps/map.h"
#endif

OnroadWindow::OnroadWindow(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout  = new QVBoxLayout(this);
  main_layout->setMargin(bdr_s);
  QStackedLayout *stacked_layout = new QStackedLayout;
  stacked_layout->setStackingMode(QStackedLayout::StackAll);
  main_layout->addLayout(stacked_layout);

  nvg = new NvgWindow(VISION_STREAM_RGB_BACK, this);

  QWidget * split_wrapper = new QWidget;
  split = new QHBoxLayout(split_wrapper);
  split->setContentsMargins(0, 0, 0, 0);
  split->setSpacing(0);
  split->addWidget(nvg);

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

void OnroadWindow::mousePressEvent(QMouseEvent* e) {
  if (map != nullptr) {
    bool sidebarVisible = geometry().x() > 0;
    map->setVisible(!sidebarVisible && !map->isVisible());
  }
  // propagation event to parent(HomeWindow)
  QWidget::mousePressEvent(e);
}

void OnroadWindow::offroadTransition(bool offroad) {
#ifdef ENABLE_MAPS
  if (!offroad) {
    if (map == nullptr && (QUIState::ui_state.has_prime || !MAPBOX_TOKEN.isEmpty())) {
      QMapboxGLSettings settings;

      // Valid for 4 weeks since we can't swap tokens on the fly
      QString token = MAPBOX_TOKEN.isEmpty() ? CommaApi::create_jwt({}, 4 * 7 * 24 * 3600) : MAPBOX_TOKEN;

      if (!Hardware::PC()) {
        settings.setCacheDatabasePath("/data/mbgl-cache.db");
      }
      settings.setApiBaseUrl(MAPS_HOST);
      settings.setCacheDatabaseMaximumSize(20 * 1024 * 1024);
      settings.setAccessToken(token.trimmed());

      MapWindow * m = new MapWindow(settings);
      m->setFixedWidth(topWidget(this)->width() / 2);
      QObject::connect(this, &OnroadWindow::offroadTransitionSignal, m, &MapWindow::offroadTransition);
      split->addWidget(m, 0, Qt::AlignRight);
      map = m;
    }
  }
#endif

  alerts->updateAlert({}, bg);

  // update stream type
  bool wide_cam = Hardware::TICI() && Params().getBool("EnableWideCamera");
  nvg->setStreamType(wide_cam ? VISION_STREAM_RGB_WIDE : VISION_STREAM_RGB_BACK);
}

void OnroadWindow::paintEvent(QPaintEvent *event) {
  QPainter p(this);
  p.fillRect(rect(), QColor(bg.red(), bg.green(), bg.blue(), 255));
}

// ***** onroad widgets *****

void OnroadAlerts::updateAlert(const Alert &a, const QColor &color) {
  if (!alert.equal(a) || color != bg) {
    alert = a;
    bg = color;
    update();
  }
}

struct Rects
{
  int x, y, w, h;     // bounding box for total text area
  int x1, y1, w1, h1; // bounding box for alert text 1
  int x2, y2, w2, h2; // bounding box for alert text 2

  Rects()
  {
    memset(this, 0, sizeof(Rects));
  }
};

template<typename T>
void getRects(T t, Rects &r, Alert alert) {
  // returns bounding boxes for text area, alert1, alert2
  // w - width, h - height, windows starting position x, y

  const int heightMax = 1080; // adjust to change text box scale
  const int width = t->width();
  const int height = t->height();

  switch (alert.size) {
    case cereal::ControlsState::AlertSize::SMALL:
      r.h = height * (271.0 / heightMax);
      break;
    case cereal::ControlsState::AlertSize::MID:
      r.h = height * (420.0 / heightMax);
      break;
    case cereal::ControlsState::AlertSize::FULL:
      r.h = height;
      break;
    case cereal::ControlsState::AlertSize::NONE:
      break;
  }

  r.w = width;
  r.w1 = r.w;
  r.w2 = r.w1;

  r.h1 = r.h / 2.0;
  r.h2 = r.h1;

  r.y = height - r.h;
  r.y1 = r.y;
  r.y2 = r.y1 + r.h1;
}

void OnroadAlerts::paintEvent(QPaintEvent *event) {
  if (alert.size == cereal::ControlsState::AlertSize::NONE) {
    return;
  }
  Rects R;
  getRects(this, R, alert);
  QRect r = QRect(R.x, R.y, R.w, R.h);
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
  p.setPen(QColor(0xff, 0xff, 0xff));
  p.setRenderHint(QPainter::TextAntialiasing);

  const QRect a1 = QRect(R.x1, R.y1, R.w1, R.h1);
  const QRect a2 = QRect(R.x2, R.y2, R.w2, R.h2);

  const int textBoxFlag = Qt::AlignCenter;

  if (alert.size == cereal::ControlsState::AlertSize::SMALL) {
    configFont(p, r, "Open Sans", 74, "SemiBold", alert.text1);
    p.drawText(r, textBoxFlag, alert.text1);
  } else if (alert.size == cereal::ControlsState::AlertSize::MID) {
    configFont(p, a1, "Open Sans", 88, "Bold", alert.text1);
    p.drawText(a1, textBoxFlag, alert.text1);
    configFont(p, a2, "Open Sans", 66, "Regular", alert.text2);
    p.drawText(a2, textBoxFlag, alert.text2);
  } else if (alert.size == cereal::ControlsState::AlertSize::FULL) {
    bool l = alert.text1.length() > 15;
    configFont(p, a1, "Open Sans", l ? 132 : 177, "Bold", alert.text1);
    p.drawText(a1, textBoxFlag, alert.text1);
    configFont(p, a2, "Open Sans", 88, "Regular", alert.text2);
    p.drawText(a2, textBoxFlag, alert.text2);
  }
}

void NvgWindow::initializeGL() {
  CameraViewWidget::initializeGL();
  qInfo() << "OpenGL version:" << QString((const char*)glGetString(GL_VERSION));
  qInfo() << "OpenGL vendor:" << QString((const char*)glGetString(GL_VENDOR));
  qInfo() << "OpenGL renderer:" << QString((const char*)glGetString(GL_RENDERER));
  qInfo() << "OpenGL language version:" << QString((const char*)glGetString(GL_SHADING_LANGUAGE_VERSION));

  ui_nvg_init(&QUIState::ui_state);
  prev_draw_t = millis_since_boot();
  setBackgroundColor(bg_colors[STATUS_DISENGAGED]);
}

void NvgWindow::paintGL() {
  CameraViewWidget::paintGL();
  ui_draw(&QUIState::ui_state, width(), height());

  double cur_draw_t = millis_since_boot();
  double dt = cur_draw_t - prev_draw_t;
  if (dt > 66) {
    // warn on sub 15fps
    LOGW("slow frame time: %.2f", dt);
  }
  prev_draw_t = cur_draw_t;
}
