#include "selfdrive/ui/qt/onroad/alerts.h"

#include <QPainter>
#include <map>

#include "selfdrive/ui/qt/util.h"

OnroadAlerts::OnroadAlerts(QWidget *parent) : QWidget(parent) {
  progressAnimation = new QPropertyAnimation(this, "currentProgress");
  connect(this, SIGNAL(currentProgressChanged(float)), this, SLOT(update()));

  delayTimer = new QTimer(this);
  delayTimer->setSingleShot(true);
  connect(delayTimer, &QTimer::timeout, this, &OnroadAlerts::startAnimationAfterDelay);
}

void OnroadAlerts::updateState(const UIState &s) {
  Alert a = getAlert(*(s.sm), s.scene.started_frame);
  if (!alert.equal(a)) {
    alert = a;
    progressAnimation->stop();
    delayTimer->stop();

    if (alert.fastForward) {
      progressAnimation->setStartValue(currentProgress);
      progressAnimation->setEndValue(1.0f);
      progressAnimation->setDuration(alert.durationMs);  // Rapid fill over alert duration
      progressAnimation->setEasingCurve(QEasingCurve::OutCubic);
      progressAnimation->start();
    } else if (alert.progressDurationMs > 0) {
      int delayMs = alert.durationMs - alert.progressDurationMs;
      float initialProgress = 0.0f;
      int remainingDuration = alert.progressDurationMs;

      if (delayMs < 0) {
        initialProgress = 1.0f - (static_cast<float>(alert.durationMs) / alert.progressDurationMs);
        initialProgress = qBound(0.0f, initialProgress, 1.0f);
        delayMs = 0;
        remainingDuration = alert.durationMs;
      }

      currentProgress = initialProgress;
      progressAnimation->setStartValue(initialProgress);
      progressAnimation->setEndValue(1.0f);
      progressAnimation->setEasingCurve(QEasingCurve::Linear);
      progressAnimation->setDuration(remainingDuration);

      if (delayMs > 0) {
        delayTimer->start(delayMs);
      } else {
        startAnimationAfterDelay();
      }
    } else {
      currentProgress = 0.0f;
    }
    update();
  }
}

void OnroadAlerts::startAnimationAfterDelay() {
  progressAnimation->start();
}

void OnroadAlerts::clear() {
  alert = {};
  delayTimer->stop();
  progressAnimation->stop();
  currentProgress = 0.0f;
  update();
}

OnroadAlerts::Alert OnroadAlerts::getAlert(const SubMaster &sm, uint64_t started_frame) {
  const cereal::SelfdriveState::Reader &ss = sm["selfdriveState"].getSelfdriveState();
  const uint64_t selfdrive_frame = sm.rcv_frame("selfdriveState");

  Alert a = {};
  if (selfdrive_frame >= started_frame) {  // Don't get old alert.
    a = {ss.getAlertText1().cStr(), ss.getAlertText2().cStr(),
         ss.getAlertType().cStr(), ss.getAlertSize(), ss.getAlertStatus(),
         ss.getAlertDuration() * 10, ss.getAlertProgressDuration() * 10, ss.getAlertFastForward()};
  }

  if (!sm.updated("selfdriveState") && (sm.frame - started_frame) > 5 * UI_FREQ) {
    const int SELFDRIVE_STATE_TIMEOUT = 5;
    const int ss_missing = (nanos_since_boot() - sm.rcv_time("selfdriveState")) / 1e9;

    // Handle selfdrive timeout
    if (selfdrive_frame < started_frame) {
      // car is started, but selfdriveState hasn't been seen at all
      a = {tr("openpilot Unavailable"), tr("Waiting to start"),
           "selfdriveWaiting", cereal::SelfdriveState::AlertSize::MID,
           cereal::SelfdriveState::AlertStatus::NORMAL};
    } else if (ss_missing > SELFDRIVE_STATE_TIMEOUT && !Hardware::PC()) {
      // car is started, but selfdrive is lagging or died
      if (ss.getEnabled() && (ss_missing - SELFDRIVE_STATE_TIMEOUT) < 10) {
        a = {tr("TAKE CONTROL IMMEDIATELY"), tr("System Unresponsive"),
             "selfdriveUnresponsive", cereal::SelfdriveState::AlertSize::FULL,
             cereal::SelfdriveState::AlertStatus::CRITICAL};
      } else {
        a = {tr("System Unresponsive"), tr("Reboot Device"),
             "selfdriveUnresponsivePermanent", cereal::SelfdriveState::AlertSize::MID,
             cereal::SelfdriveState::AlertStatus::NORMAL};
      }
    }
  }
  return a;
}

void OnroadAlerts::paintEvent(QPaintEvent *event) {
  if (alert.size == cereal::SelfdriveState::AlertSize::NONE) {
    return;
  }
  static std::map<cereal::SelfdriveState::AlertSize, const int> alert_heights = {
    {cereal::SelfdriveState::AlertSize::SMALL, 271},
    {cereal::SelfdriveState::AlertSize::MID, 420},
    {cereal::SelfdriveState::AlertSize::FULL, height()},
  };
  int h = alert_heights[alert.size];

  int margin = 40;
  int radius = 30;
  if (alert.size == cereal::SelfdriveState::AlertSize::FULL) {
    margin = 0;
    radius = 0;
  }
  QRect r = QRect(0 + margin, height() - h + margin, width() - margin*2, h - margin*2);

  QPainter p(this);

  // draw background + gradient
  p.setPen(Qt::NoPen);
  p.setCompositionMode(QPainter::CompositionMode_SourceOver);
  p.setBrush(QBrush(alert_colors[alert.status]));
  p.drawRoundedRect(r, radius, radius);

  QLinearGradient g(0, r.y(), 0, r.bottom());
  g.setColorAt(0, QColor::fromRgbF(0, 0, 0, 0.05));
  g.setColorAt(1, QColor::fromRgbF(0, 0, 0, 0.35));

  p.setCompositionMode(QPainter::CompositionMode_DestinationOver);
  p.setBrush(QBrush(g));
  p.drawRoundedRect(r, radius, radius);
  p.setCompositionMode(QPainter::CompositionMode_SourceOver);

  if (currentProgress > 0.0f) {
    const int barHeight = 16;
    const int barYOffset = r.bottom() - barHeight;
    QPainterPath clipPath;
    clipPath.addRoundedRect(r, radius, radius);
    p.setClipPath(clipPath);
    int filled = static_cast<int>(r.width() * currentProgress);
    p.setBrush(QColor(0x40, 0x40, 0x40, 0x80));
    p.drawRect(r.x(), barYOffset, r.width(), barHeight);
    p.setBrush(QColor(0x00, 0xFF, 0x00, 0xC0));
    p.drawRect(r.x(), barYOffset, filled, barHeight);
    p.setClipping(false);
  }

  // text
  const QPoint c = r.center();
  p.setPen(QColor(0xff, 0xff, 0xff));
  p.setRenderHint(QPainter::TextAntialiasing);
  if (alert.size == cereal::SelfdriveState::AlertSize::SMALL) {
    p.setFont(InterFont(74, QFont::DemiBold));
    p.drawText(r, Qt::AlignCenter, alert.text1);
  } else if (alert.size == cereal::SelfdriveState::AlertSize::MID) {
    p.setFont(InterFont(88, QFont::Bold));
    p.drawText(QRect(0, c.y() - 125, width(), 150), Qt::AlignHCenter | Qt::AlignTop, alert.text1);
    p.setFont(InterFont(66));
    p.drawText(QRect(0, c.y() + 21, width(), 90), Qt::AlignHCenter, alert.text2);
  } else if (alert.size == cereal::SelfdriveState::AlertSize::FULL) {
    bool l = alert.text1.length() > 15;
    p.setFont(InterFont(l ? 132 : 177, QFont::Bold));
    p.drawText(QRect(0, r.y() + (l ? 240 : 270), width(), 600), Qt::AlignHCenter | Qt::TextWordWrap, alert.text1);
    p.setFont(InterFont(88));
    p.drawText(QRect(0, r.height() - (l ? 361 : 420), width(), 300), Qt::AlignHCenter | Qt::TextWordWrap, alert.text2);
  }
}
