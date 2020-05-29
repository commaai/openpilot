
#include "sound.hpp"
#include <assert.h>
#include <getopt.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "common/swaglog.h"
#include "common/timing.h"

#define CHECK_RESULT(func, msg)      \
  if ((func) != SL_RESULT_SUCCESS) { LOGW(msg); return false; } \

struct Sound::sound_player {
  SLObjectItf player;
  SLPlayItf playInterface;
  bool loop;
};

struct sound_file {
  AudibleAlert alert;
  const char* uri;
  bool loop;
};

sound_file sound_table[] = {
    {cereal::CarControl::HUDControl::AudibleAlert::CHIME_DISENGAGE, "../assets/sounds/disengaged.wav", false},
    {cereal::CarControl::HUDControl::AudibleAlert::CHIME_ENGAGE, "../assets/sounds/engaged.wav", false},
    {cereal::CarControl::HUDControl::AudibleAlert::CHIME_WARNING1, "../assets/sounds/warning_1.wav", false},
    {cereal::CarControl::HUDControl::AudibleAlert::CHIME_WARNING2, "../assets/sounds/warning_2.wav", false},
    {cereal::CarControl::HUDControl::AudibleAlert::CHIME_WARNING2_REPEAT, "../assets/sounds/warning_2.wav", true},
    {cereal::CarControl::HUDControl::AudibleAlert::CHIME_WARNING_REPEAT, "../assets/sounds/warning_repeat.wav", true},
    {cereal::CarControl::HUDControl::AudibleAlert::CHIME_ERROR, "../assets/sounds/error.wav", false},
    {cereal::CarControl::HUDControl::AudibleAlert::CHIME_PROMPT, "../assets/sounds/error.wav", false},
    {cereal::CarControl::HUDControl::AudibleAlert::NONE, NULL, false},
};

bool Sound::init(int volumn) {
  SLEngineOption engineOptions[] = {{SL_ENGINEOPTION_THREADSAFE, SL_BOOLEAN_TRUE}};
  const SLInterfaceID ids[1] = {SL_IID_VOLUME};
  const SLboolean req[1] = {SL_BOOLEAN_FALSE};
  CHECK_RESULT(slCreateEngine(&engine_, 1, engineOptions, 0, NULL, NULL), "Failed to create OpenSL engine");
  CHECK_RESULT((*engine_)->Realize(engine_, SL_BOOLEAN_FALSE), "Failed to realize OpenSL engine");
  CHECK_RESULT((*engine_)->GetInterface(engine_, SL_IID_ENGINE, &engineInterface_), "Failed to realize OpenSL engine");
  CHECK_RESULT((*engineInterface_)->CreateOutputMix(engineInterface_, &outputMix_, 1, ids, req), "Failed to create output mix");
  CHECK_RESULT((*outputMix_)->Realize(outputMix_, SL_BOOLEAN_FALSE), "Failed to realize output mix");

  for (sound_file* s = sound_table; s->alert != cereal::CarControl::HUDControl::AudibleAlert::NONE; s++) {
    if (!create_player(s->alert, s->uri, s->loop)) {
      return false;
    }
  }
  setVolume(volumn);
  return true;
}

bool Sound::create_player(AudibleAlert alert, const char* uri, bool loop) {
  SLDataLocator_URI locUri = {SL_DATALOCATOR_URI, (SLchar*)uri};
  SLDataFormat_MIME formatMime = {SL_DATAFORMAT_MIME, NULL, SL_CONTAINERTYPE_UNSPECIFIED};
  SLDataSource audioSrc = {&locUri, &formatMime};

  SLDataLocator_OutputMix outMix = {SL_DATALOCATOR_OUTPUTMIX, outputMix_};
  SLDataSink audioSnk = {&outMix, NULL};

  SLObjectItf player = NULL;
  SLPlayItf playInterface = NULL;
  CHECK_RESULT((*engineInterface_)->CreateAudioPlayer(engineInterface_, &player, &audioSrc, &audioSnk, 0, NULL, NULL), "Failed to create audio player");
  CHECK_RESULT((*player)->Realize(player, SL_BOOLEAN_FALSE), "Failed to realize audio player");
  CHECK_RESULT((*player)->GetInterface(player, SL_IID_PLAY, &playInterface), "Failed to get player interface");
  CHECK_RESULT((*playInterface)->SetPlayState(playInterface, SL_PLAYSTATE_PAUSED), "Failed to initialize playstate to SL_PLAYSTATE_PAUSED");

  player_[alert] = new sound_player{player, playInterface, loop};
  return true;
}

void SLAPIENTRY Sound::slplay_callback(SLPlayItf playItf, void* context, SLuint32 event) {
  Sound* s = (Sound*)context;
  if (event == SL_PLAYEVENT_HEADATEND && s->loop_start_ctx_ == s->loop_start_) {
    (*playItf)->SetPlayState(playItf, SL_PLAYSTATE_STOPPED);
    (*playItf)->SetMarkerPosition(playItf, 0);
    (*playItf)->SetPlayState(playItf, SL_PLAYSTATE_PLAYING);
  }
}

bool Sound::play(AudibleAlert alert) {
  sound_player* sound = player_.at(alert);
  auto playInterface = sound->playInterface;
  if (sound->loop) {
    loop_start_ctx_ = loop_start_ = nanos_since_boot();
    CHECK_RESULT((*playInterface)->RegisterCallback(playInterface, slplay_callback, this), "Failed to register callback");
    CHECK_RESULT((*playInterface)->SetCallbackEventsMask(playInterface, SL_PLAYEVENT_HEADATEND), "Failed to set callback event mask");
  }

  // Reset the audio player
  CHECK_RESULT((*playInterface)->ClearMarkerPosition(playInterface), "Failed to clear marker position");
  uint32_t states[] = {SL_PLAYSTATE_PAUSED, SL_PLAYSTATE_STOPPED, SL_PLAYSTATE_PLAYING};
  for (auto state : states) {
    CHECK_RESULT((*playInterface)->SetPlayState(playInterface, state), "Failed to set SL_PLAYSTATE_PLAYING");
  }
  return true;
}

bool Sound::stop(AudibleAlert alert) {
  // stop a loop
  loop_start_ = 0;
  auto playInterface = player_.at(alert)->playInterface;
  CHECK_RESULT((*playInterface)->SetPlayState(playInterface, SL_PLAYSTATE_PAUSED), "Failed to set SL_PLAYSTATE_STOPPED");
  return true;
}

void Sound::setVolume(int volume, double current_time) {
  // 5 second timeout
  if (last_volume_ != volume && (current_time == 0 || (current_time - last_set_volume_time_) > 5 * (1e+9))) {
    char volume_change_cmd[64];
    snprintf(volume_change_cmd, sizeof(volume_change_cmd), "service call audio 3 i32 3 i32 %d i32 1 &", volume);
    system(volume_change_cmd);
    last_volume_ = volume;
    last_set_volume_time_ = current_time;
  }
}

Sound::~Sound() {
  for (auto& kv : player_) {
    if (kv.second->player) {
      (*(kv.second->player))->Destroy(kv.second->player);
    }
    delete kv.second;
  }
  if (outputMix_) (*outputMix_)->Destroy(outputMix_);
  if (engine_) (*engine_)->Destroy(engine_);
}
