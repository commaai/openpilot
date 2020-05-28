
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
  if ((func) != SL_RESULT_SUCCESS) { \
    LOGW(msg);                       \
    return false;                    \
  }

typedef struct {
  AudibleAlert alert;
  const char* uri;
  bool loop;
  SLObjectItf player;
  SLPlayItf playInterface;
} sound_file;

sound_file sound_table[] = {
    {cereal::CarControl::HUDControl::AudibleAlert::CHIME_DISENGAGE, "../assets/sounds/disengaged.wav", false, NULL, NULL},
    {cereal::CarControl::HUDControl::AudibleAlert::CHIME_ENGAGE, "../assets/sounds/engaged.wav", false, NULL, NULL},
    {cereal::CarControl::HUDControl::AudibleAlert::CHIME_WARNING1, "../assets/sounds/warning_1.wav", false, NULL, NULL},
    {cereal::CarControl::HUDControl::AudibleAlert::CHIME_WARNING2, "../assets/sounds/warning_2.wav", false, NULL, NULL},
    {cereal::CarControl::HUDControl::AudibleAlert::CHIME_WARNING2_REPEAT, "../assets/sounds/warning_2.wav", true, NULL, NULL},
    {cereal::CarControl::HUDControl::AudibleAlert::CHIME_WARNING_REPEAT, "../assets/sounds/warning_repeat.wav", true, NULL, NULL},
    {cereal::CarControl::HUDControl::AudibleAlert::CHIME_ERROR, "../assets/sounds/error.wav", false, NULL, NULL},
    {cereal::CarControl::HUDControl::AudibleAlert::CHIME_PROMPT, "../assets/sounds/error.wav", false, NULL, NULL},
    {cereal::CarControl::HUDControl::AudibleAlert::NONE, NULL, false, NULL, NULL},
};

sound_file* get_sound_file(AudibleAlert alert) {
  for (sound_file* s = sound_table; s->alert != cereal::CarControl::HUDControl::AudibleAlert::NONE; s++) {
    if (s->alert == alert) {
      return s;
    }
  }
  return NULL;
}

void SLAPIENTRY slplay_callback(SLPlayItf playItf, void* context, SLuint32 event) {
  Sound* s = (Sound*)context;
  if (event == SL_PLAYEVENT_HEADATEND && s->loop_start_ctx_ == s->loop_start_) {
    (*playItf)->SetPlayState(playItf, SL_PLAYSTATE_STOPPED);
    (*playItf)->SetMarkerPosition(playItf, 0);
    (*playItf)->SetPlayState(playItf, SL_PLAYSTATE_PLAYING);
  }
}

bool Sound::init() {
  SLEngineOption engineOptions[] = {{SL_ENGINEOPTION_THREADSAFE, SL_BOOLEAN_TRUE}};
  const SLInterfaceID ids[1] = {SL_IID_VOLUME};
  const SLboolean req[1] = {SL_BOOLEAN_FALSE};
  CHECK_RESULT(slCreateEngine(&engine_, 1, engineOptions, 0, NULL, NULL), "Failed to create OpenSL engine");
  CHECK_RESULT((*engine_)->Realize(engine_, SL_BOOLEAN_FALSE), "Failed to realize OpenSL engine");
  CHECK_RESULT((*engine_)->GetInterface(engine_, SL_IID_ENGINE, &engineInterface_), "Failed to realize OpenSL engine");
  CHECK_RESULT((*engineInterface_)->CreateOutputMix(engineInterface_, &outputMix_, 1, ids, req), "Failed to create output mix");
  CHECK_RESULT((*outputMix_)->Realize(outputMix_, SL_BOOLEAN_FALSE), "Failed to realize output mix");
  for (sound_file* s = sound_table; s->alert != cereal::CarControl::HUDControl::AudibleAlert::NONE; s++) {
    if (!slplay_create_player_for_uri(s, &error)) {
      return false;
    }
  }
  return true;
}

bool Sound::create_player_for_uri(soud_file* s) {
  SLDataLocator_URI locUri = {SL_DATALOCATOR_URI, (SLchar*)s->uri};
  SLDataFormat_MIME formatMime = {SL_DATAFORMAT_MIME, NULL, SL_CONTAINERTYPE_UNSPECIFIED};
  SLDataSource audioSrc = {&locUri, &formatMime};

  SLDataLocator_OutputMix outMix = {SL_DATALOCATOR_OUTPUTMIX, outputMix_};
  SLDataSink audioSnk = {&outMix, NULL};

  CHECK_RESULT((*engineInterface_)->CreateAudioPlayer(engineInterface_, &s->player, &audioSrc, &audioSnk, 0, NULL, NULL), "Failed to create audio player");
  CHECK_RESULT((*(s->player))->Realize(s->player, SL_BOOLEAN_FALSE), "Failed to realize audio player");
  CHECK_RESULT((*(s->player))->GetInterface(s->player, SL_IID_PLAY, &(s->playInterface)), "Failed to get player interface");
  CHECK_RESULT((*(s->playInterface))->SetPlayState(s->playInterface, SL_PLAYSTATE_PAUSED), "Failed to initialize playstate to SL_PLAYSTATE_PAUSED");
  return true;
}

bool Sound::play(AudibleAlert alert) {
  sound_file* sound = get_sound_file(alert);
  SLPlayItf playInterface = sound->playInterface;
  if (sound->loop) {
    loop_start_ = nanos_since_boot();
    loop_start_ctx_ = loop_start_;
    CHECK_RESULT((*playInterface)->RegisterCallback(playInterface, slplay_callback, this), "Failed to register callback");
    CHECK_RESULT((*playInterface)->SetCallbackEventsMask(playInterface, SL_PLAYEVENT_HEADATEND), "Failed to set callback event mask");
  }

  // Reset the audio player
  CHECK_RESULT((*playInterface)->ClearMarkerPosition(playInterface), "Failed to clear marker position");

  auto setState = [&playInterface](auto state) {
    CHECK_RESULT((*playInterface)->SetPlayState(playInterface, state), "Failed to set SL_PLAYSTATE_PLAYING");
  };
  setState(SL_PLAYSTATE_PAUSED);
  setState(SL_PLAYSTATE_STOPPED);
  setState(SL_PLAYSTATE_PLAYING);
  return true;
}

bool Sound::stop(AudibleAlert alert) {
  sound_file* sound = get_sound_file(alert);
  SLPlayItf playInterface = sound->playInterface;
  // stop a loop
  loop_start_ = 0;
  CHECK_RESULT((*playInterface)->SetPlayState(playInterface, SL_PLAYSTATE_PAUSED), "Failed to set SL_PLAYSTATE_STOPPED");
  return true;
}

void Sound::set_volume(int volume) {
  static int last_volume = 0;
  if (last_volume != volume) {
    char volume_change_cmd[64];
    snprintf(volume_change_cmd, sizeof(volume_change_cmd), "service call audio 3 i32 3 i32 %d i32 1 &", volume);

    // 5 second timeout at 60fps
    int volume_changed = system(volume_change_cmd);
    last_volume = volume;
  }
}

void Sound::destroy() {
  for (sound_file* s = sound_table; s->alert != cereal::CarControl::HUDControl::AudibleAlert::NONE; s++) {
    if (s->player) {
      (*(s->player))->Destroy(s->player);
      s->player = NULL;
    }
  }
  if (outputMix_) {
    (*outputMix_)->Destroy(outputMix_);
    outputMix_ = NULL;
  }
  if (engine_) {
    (*engine_)->Destroy(engine_);
    engine_ = NULL;
  }
}

Sound::~Sound() {
  destroy();
}