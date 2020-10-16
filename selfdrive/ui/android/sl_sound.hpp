#pragma once
#include <SLES/OpenSLES.h>
#include <SLES/OpenSLES_Android.h>

#include "sound.hpp"


class SLSound : public Sound {
public:
  SLSound();
  ~SLSound();
  bool play(AudibleAlert alert);
  void stop();
  void setVolume(int volume);

private:
  bool init();
  SLObjectItf engine_ = nullptr;
  SLObjectItf outputMix_ = nullptr;
  int last_volume_ = 0;
  double last_set_volume_time_ = 0.;
  AudibleAlert currentSound_ = AudibleAlert::NONE;
  struct Player;
  std::map<AudibleAlert, Player *> player_;
  friend void SLAPIENTRY slplay_callback(SLPlayItf playItf, void *context, SLuint32 event);
};
