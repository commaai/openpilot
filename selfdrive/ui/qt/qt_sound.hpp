#pragma once

#include <QSoundEffect>
#include "sound.hpp"

class QtSound : public Sound {
public:
  QtSound();
  bool play(AudibleAlert alert);
  void stop();
  void setVolume(int volume);
  float volume = 0;

private:
  std::map<AudibleAlert, QSoundEffect> sounds;
};
