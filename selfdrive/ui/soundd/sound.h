#pragma once

#include <QMap>
#include <QSoundEffect>
#include <QString>

#include "selfdrive/ui/ui.h"

struct SoundItem {
  const char *file;
  int loops;
  int loops_to_full_volume;
  QSoundEffect *sound;
};

class Sound : public QObject {
public:
  explicit Sound(QObject *parent = 0);

protected:
  void update();
  void updateVolume(const SoundItem &s);
  void setAlert(const Alert &alert);

  const int LOOP_INFINITE = std::numeric_limits<int>::max();

  QMap<AudibleAlert, SoundItem> sounds = {
    {AudibleAlert::ENGAGE, {"engage.wav", 0, 0}},
    {AudibleAlert::DISENGAGE, {"disengage.wav", 0, 0}},
    {AudibleAlert::REFUSE, {"refuse.wav", 0, 0}},

    {AudibleAlert::PROMPT, {"prompt.wav", 0, 0}},
    {AudibleAlert::PROMPT_REPEAT, {"prompt.wav", LOOP_INFINITE, 0}},
    {AudibleAlert::PROMPT_DISTRACTED, {"prompt_distracted.wav", LOOP_INFINITE, 6}},

    {AudibleAlert::WARNING_SOFT, {"warning_soft.wav", LOOP_INFINITE, 0}},
    {AudibleAlert::WARNING_IMMEDIATE, {"warning_immediate.wav", QSoundEffect::Infinite, 6}},
  };

  Alert current_alert = {};
  qreal current_volume;
  SubMaster sm;
  uint64_t started_frame = 0;
  bool started_prev = false;
};
