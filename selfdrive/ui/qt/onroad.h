#pragma once
#include <map>

#include <QSoundEffect>
#include <QtWidgets>

#include "cereal/gen/cpp/log.capnp.h"
#include "selfdrive/hardware/hw.h"
#include "selfdrive/ui/ui.h"


typedef cereal::CarControl::HUDControl::AudibleAlert AudibleAlert;

// ***** onroad widgets *****

class OnroadAlerts : public QFrame {
  Q_OBJECT

public:
  OnroadAlerts(QWidget *parent = 0);

protected:
  void paintEvent(QPaintEvent*) override;

private:
  QColor bg;
  QLabel *title, *msg;
  QVBoxLayout *layout;

  void updateAlert(const QString &text1, const QString &text2, float blink_rate,
                   const std::string &type, cereal::ControlsState::AlertSize size, AudibleAlert sound);

  // sounds
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
  float volume = Hardware::MIN_VOLUME;
  float blinking_rate = 0;
  std::string alert_type;
  std::map<AudibleAlert, QSoundEffect> sounds;

  void playSound(AudibleAlert alert);
  void stopSounds();

public slots:
  void update(const UIState &s);
  void offroadTransition(bool offroad);
};

// container window for the NVG UI
class NvgWindow : public QOpenGLWidget, protected QOpenGLFunctions {
  Q_OBJECT

public:
  using QOpenGLWidget::QOpenGLWidget;
  explicit NvgWindow(QWidget* parent = 0) : QOpenGLWidget(parent) {};
  ~NvgWindow();

protected:
  void paintGL() override;
  void initializeGL() override;

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

private:
  OnroadAlerts *alerts;
  NvgWindow *nvg;
  QStackedLayout *layout;

signals:
  void update(const UIState &s);
  void offroadTransition(bool offroad);
};
