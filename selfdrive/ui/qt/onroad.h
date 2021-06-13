#pragma once

#include <map>

#include <QSoundEffect>
#include <QtWidgets>

#include "cereal/gen/cpp/log.capnp.h"
#include "selfdrive/hardware/hw.h"
#include "selfdrive/ui/ui.h"
#include "selfdrive/ui/qt/qt_window.h"

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

  std::map<AudibleAlert, std::pair<QString, bool>> sound_map {
    // AudibleAlert, (file path, inf loop)
    {AudibleAlert::CHIME_DISENGAGE, {"../assets/sounds/disengaged.wav", false}},
    {AudibleAlert::CHIME_ENGAGE, {"../assets/sounds/engaged.wav", false}},
    {AudibleAlert::CHIME_WARNING1, {"../assets/sounds/warning_1.wav", false}},
    {AudibleAlert::CHIME_WARNING2, {"../assets/sounds/warning_2.wav", false}},
    {AudibleAlert::CHIME_WARNING2_REPEAT, {"../assets/sounds/warning_2.wav", true}},
    {AudibleAlert::CHIME_WARNING_REPEAT, {"../assets/sounds/warning_repeat.wav", true}},
    {AudibleAlert::CHIME_ERROR, {"../assets/sounds/error.wav", false}},
    {AudibleAlert::CHIME_PROMPT, {"../assets/sounds/error.wav", false}}
  };

  QColor bg;
  float volume = Hardware::MIN_VOLUME;
  std::map<AudibleAlert, QSoundEffect> sounds;
  float blinking_rate = 0;
  QString text1, text2;
  std::string alert_type;
  cereal::ControlsState::AlertSize alert_size;

public slots:
  void updateState(const UIState &s);
  void offroadTransition(bool offroad);
};

// container window for the NVG UI
class NvgWindow : public QOpenGLWidget, protected QOpenGLFunctions {
  Q_OBJECT

public:
  using QOpenGLWidget::QOpenGLWidget;
  explicit NvgWindow(QWidget* parent = 0);
  ~NvgWindow();

protected:
  void paintGL() override;
  void initializeGL() override;
  void resizeGL(int w, int h) override;

private:
  double prev_draw_t = 0;

public slots:
  void update(const UIState &s);
};

// container for all onroad widgets
class OnroadWindow : public QWidget {
  Q_OBJECT

public:
  OnroadWindow(QWidget* parent = 0);
  QWidget *map = nullptr;

private:
  OnroadAlerts *alerts;
  NvgWindow *nvg;
  QStackedLayout *main_layout;
  QHBoxLayout* split;

signals:
  void update(const UIState &s);
  void offroadTransitionSignal(bool offroad);

private slots:
  void offroadTransition(bool offroad);
};
