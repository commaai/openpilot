#pragma once
#include "cereal/gen/cpp/log.capnp.h"

typedef cereal::CarControl::HUDControl::AudibleAlert AudibleAlert;
#if defined(QCOM) || defined(QCOM2)
#include <SLES/OpenSLES.h>
#include <SLES/OpenSLES_Android.h>
#endif
class Sound {
 public:
  Sound() = default;
  bool init();
  bool play(AudibleAlert alert);
  bool stop(AudibleAlert alert);
  void set_volume(int volume, double current_time);
  ~Sound();
#if defined(QCOM) || defined(QCOM2)
 private:
  bool create_player_for_uri(soud_file* s);
  SLEngineItf engineInterface_ = NULL;
  SLObjectItf outputMix_ = NULL;
  SLObjectItf engine_ = NULL;
  uint64_t loop_start_ = 0;
  uint64_t loop_start_ctx_ = 0;
#endif
};
