#include <QMap>
#include <QSoundEffect>
#include <QString>

#include "system/hardware/hw.h"
#include "selfdrive/ui/ui.h"

const std::tuple<AudibleAlert, QString, int> sound_list[] = {
  // AudibleAlert, file name, loop count
  // {AudibleAlert::ENGAGE, "engage.wav", 0},
  // {AudibleAlert::DISENGAGE, "disengage.wav", 0},
  // {AudibleAlert::REFUSE, "refuse.wav", 0},

  // {AudibleAlert::PROMPT, "prompt.wav", 0},
  {AudibleAlert::PROMPT_REPEAT, "prompt.wav", QSoundEffect::Infinite},
  {AudibleAlert::PROMPT_DISTRACTED, "prompt_distracted.wav", QSoundEffect::Infinite},

  {AudibleAlert::WARNING_SOFT, "warning_soft.wav", QSoundEffect::Infinite},
  {AudibleAlert::WARNING_IMMEDIATE, "warning_immediate.wav", QSoundEffect::Infinite},
};

class Sound : public QObject {
public:
  explicit Sound(QObject *parent = 0);

protected:
  void update();
  void setAlert(const Alert &alert);
  void updateVolume(bool ramp_up);

  Alert current_alert = {};
  float current_start_time = 0;
  float current_volume = 0.5;
  QMap<AudibleAlert, QPair<QSoundEffect *, int>> sounds;
  SubMaster sm;
  uint64_t started_frame;
};
