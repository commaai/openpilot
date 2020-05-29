#pragma once
#include "cereal/gen/cpp/log.capnp.h"
#include <map>
typedef cereal::CarControl::HUDControl::AudibleAlert AudibleAlert;
#if defined(QCOM) || defined(QCOM2)
#include <SLES/OpenSLES.h>
#include <SLES/OpenSLES_Android.h>
#endif
class Sound {
 public:
  Sound() = default;
  bool init(int volumn);
  bool play(AudibleAlert alert);
  bool stop(AudibleAlert alert);
  void setVolume(int volume, double current_time = 0);
  ~Sound();
#if defined(QCOM) || defined(QCOM2)
 private:
  struct sound_player;
  bool create_player(AudibleAlert alert, const char* uri, bool loop);
  static void SLAPIENTRY slplay_callback(SLPlayItf playItf, void* context, SLuint32 event);
  SLEngineItf engineInterface_ = NULL;
  SLObjectItf outputMix_ = NULL;
  SLObjectItf engine_ = NULL;
  uint64_t loop_start_ = 0;
  uint64_t loop_start_ctx_ = 0;
  int last_volume_ = 0;
  double last_set_volume_time_ = 0;
  std::map<AudibleAlert, sound_player*> player_;
#endif
};
