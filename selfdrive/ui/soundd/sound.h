#pragma once

#include <tuple>

#include <QMap>
#include <QSoundEffect>
#include <QString>

#include "system/hardware/hw.h"
#include "selfdrive/ui/ui.h"


const float MAX_VOLUME = 1.0;

const std::tuple<AudibleAlert, QString, int, float> sound_list[] = {
  // AudibleAlert, file name, loop count
  {AudibleAlert::ENGAGE, "engage.wav", 0, MAX_VOLUME},
  {AudibleAlert::DISENGAGE, "disengage.wav", 0, MAX_VOLUME},
  {AudibleAlert::REFUSE, "refuse.wav", 0, MAX_VOLUME},

  {AudibleAlert::PROMPT, "prompt.wav", 0, MAX_VOLUME},
  {AudibleAlert::PROMPT_REPEAT, "prompt.wav", QSoundEffect::Infinite, MAX_VOLUME},
  {AudibleAlert::PROMPT_DISTRACTED, "prompt_distracted.wav", QSoundEffect::Infinite, MAX_VOLUME},

  {AudibleAlert::WARNING_SOFT, "warning_soft.wav", QSoundEffect::Infinite, MAX_VOLUME},
  {AudibleAlert::WARNING_IMMEDIATE, "warning_immediate.wav", QSoundEffect::Infinite, MAX_VOLUME},
};

class Sound : public QObject {
public:
  explicit Sound(QObject *parent = 0);

protected:
  void update();
  void setAlert(const Alert &alert);

  SubMaster sm;
  Alert current_alert = {};
  QMap<AudibleAlert, QPair<QSoundEffect *, int>> sounds;
  int current_volume = -1;
};
