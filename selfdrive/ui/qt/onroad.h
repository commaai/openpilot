#pragma once

#include <map>

#include <QOpenGLFunctions>
#include <QOpenGLWidget>
#include <QSoundEffect>
#include <QStackedLayout>
#include <QWidget>

#include "cereal/gen/cpp/log.capnp.h"
#include "selfdrive/hardware/hw.h"
#include "selfdrive/ui/qt/qt_window.h"
#include "selfdrive/ui/qt/widgets/cameraview.h"
#include "selfdrive/ui/ui.h"

typedef cereal::CarControl::HUDControl::AudibleAlert AudibleAlert;

// ***** onroad widgets *****

class OnroadAlerts : public QWidget {
  Q_OBJECT

public:
  OnroadAlerts(QWidget *parent = 0);

protected:
  void paintEvent(QPaintEvent*) override;

private:
  void stopSounds();
  void playSound(AudibleAlert alert);
  void updateAlert(const QString &t1, const QString &t2, float blink_rate,
                   const std::string &type, cereal::ControlsState::AlertSize size, AudibleAlert sound);

  QColor bg;
  UIStatus prev_status = STATUS_DISENGAGED;
  float volume = Hardware::MIN_VOLUME;
  std::map<AudibleAlert, std::pair<QSoundEffect, int>> sounds;
  float blinking_rate = 0;
  QString text1, text2;
  std::string alert_type;
  cereal::ControlsState::AlertSize alert_size;

public slots:
  void updateState(const UIState &s);
  void offroadTransition(bool offroad);
};

class OnroadHud : public QWidget {
  Q_OBJECT
  Q_PROPERTY(QString speed MEMBER speed_ NOTIFY valueChanged);
  Q_PROPERTY(QString speedUnit MEMBER speedUnit_ NOTIFY valueChanged);
  Q_PROPERTY(QString maxSpeed MEMBER maxSpeed_ NOTIFY valueChanged);
  Q_PROPERTY(bool engageable MEMBER engageable_ NOTIFY valueChanged);
  Q_PROPERTY(bool dmActive MEMBER dmActive_ NOTIFY valueChanged);
  Q_PROPERTY(bool hideDM MEMBER hideDM_ NOTIFY valueChanged);
  Q_PROPERTY(int status MEMBER status_ NOTIFY valueChanged);

public:
  OnroadHud(QWidget *parent = 0);
  void updateState(const UIState &s);
  void offroadTransition(bool offroad);
   const int radius = 180;
  const int img_size = 135;

protected:
  void paintEvent(QPaintEvent*) override;
  void drawIcon(QPainter &p, const QPoint &center, QPixmap &img, QBrush bg, float opacity = 1.0);
  QPixmap engage_img, dm_img;
  QString speed_, speedUnit_, maxSpeed_;
  bool metric = false, engageable_ = false, dmActive_ = false, hideDM_ = false;
  int status_ = STATUS_DISENGAGED;
  

signals:
  void valueChanged();
};

// container window for the NVG UI
class NvgWindow : public CameraViewWidget {
  Q_OBJECT

public:
  explicit NvgWindow(VisionStreamType type, QWidget* parent = 0);

protected:
  void initializeGL() override;
  void resizeGL(int w, int h) override;
  void paintGL() override;
  void showEvent(QShowEvent *event) override;
};

// container for all onroad widgets
class OnroadWindow : public QWidget {
  Q_OBJECT

public:
  OnroadWindow(QWidget* parent = 0);
  QWidget *map = nullptr;

private:
  OnroadAlerts *alerts;
  OnroadHud *scene;
  NvgWindow *nvg;
  QStackedLayout *main_layout;
  QHBoxLayout* split;

signals:
  void update(const UIState &s);
  void offroadTransitionSignal(bool offroad);

private slots:
  void offroadTransition(bool offroad);
};
