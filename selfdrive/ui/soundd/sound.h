#include <QMap>
#include <QSoundEffect>
#include <QString>

#include "selfdrive/hardware/hw.h"
#include "selfdrive/ui/ui.h"

const std::tuple<AudibleAlert, QString, int> sound_list[] = {
  // AudibleAlert, file name, loop count
  {AudibleAlert::CHIME_DISENGAGE, "disengaged.wav", 0},
  {AudibleAlert::CHIME_ENGAGE, "engaged.wav", 0},
  {AudibleAlert::CHIME_WARNING1, "warning_1.wav", 0},
  {AudibleAlert::CHIME_WARNING_REPEAT, "warning_repeat.wav", 10},
  {AudibleAlert::CHIME_WARNING_REPEAT_INFINITE, "warning_repeat.wav", QSoundEffect::Infinite},
  {AudibleAlert::CHIME_WARNING2_REPEAT_INFINITE, "warning_2.wav", QSoundEffect::Infinite},
  {AudibleAlert::CHIME_ERROR, "error.wav", 0},
  {AudibleAlert::CHIME_PROMPT, "error.wav", 0},
};

class Sound : public QObject {
public:
  explicit Sound(QObject *parent = 0);

protected:
  void update();
  void setAlert(const Alert &alert);

  Alert current_alert = {};
  QMap<AudibleAlert, QPair<QSoundEffect *, int>> sounds;
  SubMaster sm;
};
