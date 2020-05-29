#pragma once
#include <map>
#include "cereal/gen/cpp/log.capnp.h"
typedef cereal::CarControl::HUDControl::AudibleAlert AudibleAlert;
#if defined(QCOM) || defined(QCOM2)
#include <SLES/OpenSLES.h>
#include <SLES/OpenSLES_Android.h>
#endif
class Sound {
 public:
  Sound() = default;
  bool init(int volumn);
  bool play(AudibleAlert alert, int repeat = 0);
  bool stop();
  void setVolume(int volume, double current_time = 0);
  inline AudibleAlert currentSound() const { return currentSound_; }
  ~Sound();
#if defined(QCOM) || defined(QCOM2)
 private:
  struct sound_player;
  bool create_player(AudibleAlert alert, const char* uri, bool loop);
  static void SLAPIENTRY slplay_callback(SLPlayItf playItf, void* context, SLuint32 event);
  SLEngineItf engineInterface_ = NULL;
  SLObjectItf outputMix_ = NULL;
  SLObjectItf engine_ = NULL;
  int repeat_ = 0;
  std::map<AudibleAlert, sound_player*> player_;
  AudibleAlert currentSound_ = cereal::CarControl::HUDControl::AudibleAlert::NONE;
#endif
};
