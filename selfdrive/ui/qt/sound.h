#pragma once

#include <map>
#include <QSoundEffect>
#include "cereal/gen/cpp/log.capnp.h"

typedef cereal::CarControl::HUDControl::AudibleAlert AudibleAlert;

class Sound {
public:
  Sound();
  void play(AudibleAlert alert);
  void stop();
  float volume = 0;

private:
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

  std::map<AudibleAlert, QSoundEffect> sounds;
};
