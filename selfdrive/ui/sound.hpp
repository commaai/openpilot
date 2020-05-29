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
  void setVolume(int volume, int timeout_seconds = 0);
  inline AudibleAlert currentSound() const { return currentSound_; }
  ~Sound();
#if defined(QCOM) || defined(QCOM2)
  struct Player;

 private:
  bool createPlayer(SLEngineItf engineInterface, AudibleAlert alert, const char* uri);
  SLObjectItf engine_ = NULL;
  SLObjectItf outputMix_ = NULL;
  int repeat_ = 0, last_volume_ = 0;
  double last_set_volume_time_ = 0;
  std::map<AudibleAlert, Player*> player_;
#endif
  AudibleAlert currentSound_ = AudibleAlert::NONE;
};
