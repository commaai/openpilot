
#include "sound.hpp"
#include <assert.h>
#include <getopt.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "common/swaglog.h"
#include "common/timing.h"

#define ReturnOnError(func, msg) \
  if ((func) != SL_RESULT_SUCCESS) { LOGW(msg); return false; } \

struct Sound::Player {
  SLObjectItf player;
  SLPlayItf playInterface;
};

struct sound_file {
  AudibleAlert alert;
  const char* uri;
};

sound_file sound_table[] = {
    {cereal::CarControl::HUDControl::AudibleAlert::CHIME_DISENGAGE, "../assets/sounds/disengaged.wav"},
    {cereal::CarControl::HUDControl::AudibleAlert::CHIME_ENGAGE, "../assets/sounds/engaged.wav"},
    {cereal::CarControl::HUDControl::AudibleAlert::CHIME_WARNING1, "../assets/sounds/warning_1.wav"},
    {cereal::CarControl::HUDControl::AudibleAlert::CHIME_WARNING2, "../assets/sounds/warning_2.wav"},
    {cereal::CarControl::HUDControl::AudibleAlert::CHIME_WARNING2_REPEAT, "../assets/sounds/warning_2.wav"},
    {cereal::CarControl::HUDControl::AudibleAlert::CHIME_WARNING_REPEAT, "../assets/sounds/warning_repeat.wav"},
    {cereal::CarControl::HUDControl::AudibleAlert::CHIME_ERROR, "../assets/sounds/error.wav"},
    {cereal::CarControl::HUDControl::AudibleAlert::CHIME_PROMPT, "../assets/sounds/error.wav"},
    {cereal::CarControl::HUDControl::AudibleAlert::NONE, nullptr},
};

bool Sound::createPlayer(SLEngineItf engineInterface, AudibleAlert alert, const char* uri) {
  SLDataLocator_URI locUri = {SL_DATALOCATOR_URI, (SLchar*)uri};
  SLDataFormat_MIME formatMime = {SL_DATAFORMAT_MIME, NULL, SL_CONTAINERTYPE_UNSPECIFIED};
  SLDataSource audioSrc = {&locUri, &formatMime};

  SLDataLocator_OutputMix outMix = {SL_DATALOCATOR_OUTPUTMIX, outputMix_};
  SLDataSink audioSnk = {&outMix, NULL};

  SLObjectItf player = NULL;
  SLPlayItf playInterface = NULL;
  ReturnOnError((*engineInterface)->CreateAudioPlayer(engineInterface, &player, &audioSrc, &audioSnk, 0, NULL, NULL), "Failed to create audio player");
  ReturnOnError((*player)->Realize(player, SL_BOOLEAN_FALSE), "Failed to realize audio player");
  ReturnOnError((*player)->GetInterface(player, SL_IID_PLAY, &playInterface), "Failed to get player interface");
  ReturnOnError((*playInterface)->SetPlayState(playInterface, SL_PLAYSTATE_PAUSED), "Failed to initialize playstate to SL_PLAYSTATE_PAUSED");

  player_[alert] = new Sound::Player{player, playInterface};
  return true;
}

bool Sound::init(int volumn) {
  SLEngineOption engineOptions[] = {{SL_ENGINEOPTION_THREADSAFE, SL_BOOLEAN_TRUE}};
  const SLInterfaceID ids[1] = {SL_IID_VOLUME};
  const SLboolean req[1] = {SL_BOOLEAN_FALSE};
  SLEngineItf engineInterface = NULL;
  ReturnOnError(slCreateEngine(&engine_, 1, engineOptions, 0, NULL, NULL), "Failed to create OpenSL engine");
  ReturnOnError((*engine_)->Realize(engine_, SL_BOOLEAN_FALSE), "Failed to realize OpenSL engine");
  ReturnOnError((*engine_)->GetInterface(engine_, SL_IID_ENGINE, &engineInterface), "Failed to realize OpenSL engine");
  ReturnOnError((*engineInterface)->CreateOutputMix(engineInterface, &outputMix_, 1, ids, req), "Failed to create output mix");
  ReturnOnError((*outputMix_)->Realize(outputMix_, SL_BOOLEAN_FALSE), "Failed to realize output mix");

  for (sound_file* s = sound_table; s->uri != nullptr; s++) {
    if (!createPlayer(engineInterface, s->alert, s->uri)) {
      return false;
    }
  }
  setVolume(volumn);
  return true;
}

void SLAPIENTRY slplay_callback(SLPlayItf playItf, void* context, SLuint32 event) {
  int *repeat  = (int*)context;
  if (event == SL_PLAYEVENT_HEADATEND && --(*repeat) > 0) {
    (*playItf)->SetPlayState(playItf, SL_PLAYSTATE_STOPPED);
    (*playItf)->SetMarkerPosition(playItf, 0);
    (*playItf)->SetPlayState(playItf, SL_PLAYSTATE_PLAYING);
  }
}

bool Sound::play(AudibleAlert alert, int repeat) {
  stop();
  currentSound_ = alert;
  auto playerItf = player_.at(alert)->playInterface;
  if (repeat > 0) {
    repeat_ = repeat;
    ReturnOnError((*playerItf)->RegisterCallback(playerItf, slplay_callback, &repeat_), "Failed to register callback");
    ReturnOnError((*playerItf)->SetCallbackEventsMask(playerItf, SL_PLAYEVENT_HEADATEND), "Failed to set callback event mask");
  }

  // Reset the audio player
  ReturnOnError((*playerItf)->ClearMarkerPosition(playerItf), "Failed to clear marker position");
  uint32_t states[] = {SL_PLAYSTATE_PAUSED, SL_PLAYSTATE_STOPPED, SL_PLAYSTATE_PLAYING};
  for (auto state : states) {
    ReturnOnError((*playerItf)->SetPlayState(playerItf, state), "Failed to set SL_PLAYSTATE_PLAYING");
  }
  return true;
}

bool Sound::stop() {
  // stop a loop
  repeat_ = 0;
  if (currentSound_ != cereal::CarControl::HUDControl::AudibleAlert::NONE) {
    currentSound_ = cereal::CarControl::HUDControl::AudibleAlert::NONE;
    auto playerItf = player_.at(currentSound_)->playInterface;
    ReturnOnError((*playerItf)->SetPlayState(playerItf, SL_PLAYSTATE_PAUSED), "Failed to set SL_PLAYSTATE_STOPPED");
  }
  return true;
}

void Sound::setVolume(int volume, double current_time) {
  static int last_volume = 0;
  static double last_set_volume_time = 0;
  // 5 second timeout
  if (last_volume != volume && (current_time == 0 || (current_time - last_set_volume_time) > 5 * (1e+9))) {
    char volume_change_cmd[64];
    snprintf(volume_change_cmd, sizeof(volume_change_cmd), "service call audio 3 i32 3 i32 %d i32 1 &", volume);
    system(volume_change_cmd);
    last_volume = volume;
    last_set_volume_time = current_time;
  }
}

Sound::~Sound() {
  for (auto& kv : player_) {
    (*(kv.second->player))->Destroy(kv.second->player);
    delete kv.second;
  }
  if (outputMix_) (*outputMix_)->Destroy(outputMix_);
  if (engine_) (*engine_)->Destroy(engine_);
}
