#pragma once

#include <QMap>
#include <QSoundEffect>
#include <QString>

#include "selfdrive/hardware/hw.h"
#include "selfdrive/ui/ui.h"

constexpr std::tuple<AudibleAlert, const char*, int, int> sound_list[] = {
  // AudibleAlert, file name, loop count, the number of loops to the full volume
  {AudibleAlert::ENGAGE, "engage.wav", 0, 0},
  {AudibleAlert::DISENGAGE, "disengage.wav", 0, 0},
  {AudibleAlert::REFUSE, "refuse.wav", 0, 0},

  {AudibleAlert::PROMPT, "prompt.wav", 0, 0},
  {AudibleAlert::PROMPT_REPEAT, "prompt.wav", QSoundEffect::Infinite, 0},
  {AudibleAlert::PROMPT_DISTRACTED, "prompt_distracted.wav", QSoundEffect::Infinite, 6},

  {AudibleAlert::WARNING_SOFT, "warning_soft.wav", QSoundEffect::Infinite, 0},
  {AudibleAlert::WARNING_IMMEDIATE, "warning_immediate.wav", 10, 6},
};

class Sound : public QObject {
public:
  explicit Sound(QObject *parent = 0);

protected:
  struct SoundItem {
    QSoundEffect *sound;
    int loops;
    int loops_to_full_volume;
  };

  void update();
  void updateVolume(const SoundItem &s);
  void setAlert(const Alert &alert);

  Alert current_alert = {};
  qreal current_volume;
  SubMaster sm;
  QMap<AudibleAlert, SoundItem> sounds;
  uint64_t started_frame = 0;
  bool started_prev = false;
};
