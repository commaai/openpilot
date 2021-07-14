#pragma once

#include <map>

#include <QHBoxLayout>
#include <QOpenGLFunctions>
#include <QOpenGLWidget>
#include <QSoundEffect>

#include <QWidget>

#include "cereal/gen/cpp/log.capnp.h"
#include "selfdrive/hardware/hw.h"
#include "selfdrive/ui/qt/qt_window.h"
#include "selfdrive/ui/ui.h"

typedef cereal::CarControl::HUDControl::AudibleAlert AudibleAlert;

// ***** onroad widgets *****

class OnroadAlerts : public QWidget {
  Q_OBJECT

public:
  OnroadAlerts(QWidget *parent = 0);
  QColor updateState(const UIState &s);
  void offroadTransition(bool offroad);

protected:
  void paintEvent(QPaintEvent*) override;

private:
  void stopSounds();
  void playSound(AudibleAlert alert);
  void updateAlert(const QString &t1, const QString &t2, float blink_rate,
                   const std::string &type, cereal::ControlsState::AlertSize size, AudibleAlert sound);

  QColor bg;
  float volume = Hardware::MIN_VOLUME;
  std::map<AudibleAlert, std::pair<QSoundEffect, int>> sounds;
  float blinking_rate = 0;
  QString text1, text2;
  std::string alert_type;
  cereal::ControlsState::AlertSize alert_size = cereal::ControlsState::AlertSize::NONE;
};

// container window for the NVG UI
class NvgWindow : public QOpenGLWidget, protected QOpenGLFunctions {
  Q_OBJECT

public:
  using QOpenGLWidget::QOpenGLWidget;
  explicit NvgWindow(QWidget* parent = 0);
  void updateState(const UIState &s);
  ~NvgWindow();

protected:
  void paintGL() override;
  void initializeGL() override;
  void resizeGL(int w, int h) override;

private:
  double prev_draw_t = 0;
};

// container for all onroad widgets
class OnroadWindow : public QWidget {
  Q_OBJECT

public:
  OnroadWindow(QWidget* parent = 0);
  QWidget *map = nullptr;

public slots:
  void offroadTransition(bool offroad);
  void updateState(const UIState &s);

private:
  void paintEvent(QPaintEvent *event);

  OnroadAlerts *alerts;
  NvgWindow *nvg;
  QColor bg;
  QHBoxLayout* split;
};
