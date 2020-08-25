#pragma once
#include <map>
#include "cereal/gen/cpp/log.capnp.h"

#ifdef QCOM
#include <SLES/OpenSLES.h>
#include <SLES/OpenSLES_Android.h>
#else
#include <QSoundEffect>
#endif

typedef cereal::CarControl::HUDControl::AudibleAlert AudibleAlert;

static std::map<AudibleAlert, std::pair<const char *, int>> sound_map {
  // AudibleAlert, (file path, loop count)
  {AudibleAlert::CHIME_DISENGAGE, {"../assets/sounds/disengaged.wav", 0}},
  {AudibleAlert::CHIME_ENGAGE, {"../assets/sounds/engaged.wav", 0}},
  {AudibleAlert::CHIME_WARNING1, {"../assets/sounds/warning_1.wav", 0}},
  {AudibleAlert::CHIME_WARNING2, {"../assets/sounds/warning_2.wav", 0}},
  {AudibleAlert::CHIME_WARNING2_REPEAT, {"../assets/sounds/warning_2.wav", 3}},
  {AudibleAlert::CHIME_WARNING_REPEAT, {"../assets/sounds/warning_repeat.wav", 3}},
  {AudibleAlert::CHIME_ERROR, {"../assets/sounds/error.wav", 0}},
  {AudibleAlert::CHIME_PROMPT, {"../assets/sounds/error.wav", 0}}
};

class Sound {
 public:
  Sound() = default;
  bool init(int volume);
  bool play(AudibleAlert alert);
  void stop();
  void setVolume(int volume);
  ~Sound();

 private:
#ifdef QCOM
  SLObjectItf engine_ = nullptr;
  SLObjectItf outputMix_ = nullptr;
  int last_volume_ = 0;
  double last_set_volume_time_ = 0.;
  AudibleAlert currentSound_ = AudibleAlert::NONE;
  struct Player;
  std::map<AudibleAlert, Player *> player_;
  friend void SLAPIENTRY slplay_callback(SLPlayItf playItf, void *context, SLuint32 event);
#else
  std::map<AudibleAlert, QSoundEffect> sounds;
#endif
};
