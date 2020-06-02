
#include "sound.hpp"
#include <assert.h>
#include <stdlib.h>
#include "common/swaglog.h"
#include "common/timing.h"

#define ReturnOnError(func, msg) \
  if ((func) != SL_RESULT_SUCCESS) { LOGW(msg); return false; }

static std::map<AudibleAlert, const char*> sound_map{
  {AudibleAlert::CHIME_DISENGAGE, "../assets/sounds/disengaged.wav"},
  {AudibleAlert::CHIME_ENGAGE, "../assets/sounds/engaged.wav"},
  {AudibleAlert::CHIME_WARNING1, "../assets/sounds/warning_1.wav"},
  {AudibleAlert::CHIME_WARNING2, "../assets/sounds/warning_2.wav"},
  {AudibleAlert::CHIME_WARNING2_REPEAT, "../assets/sounds/warning_2.wav"},
  {AudibleAlert::CHIME_WARNING_REPEAT, "../assets/sounds/warning_repeat.wav"},
  {AudibleAlert::CHIME_ERROR, "../assets/sounds/error.wav"},
  {AudibleAlert::CHIME_PROMPT, "../assets/sounds/error.wav"},
};

struct Sound::Player {
  SLObjectItf player;
  SLPlayItf playInterface;
};

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

  for (auto &kv : sound_map) {
    SLDataLocator_URI locUri = {SL_DATALOCATOR_URI, (SLchar*)kv.second};
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

    player_[kv.first] = new Sound::Player{player, playInterface};
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
  repeat_ = 0;
  if (currentSound_ != AudibleAlert::NONE) {
    auto playerItf = player_.at(currentSound_)->playInterface;
    currentSound_ = AudibleAlert::NONE;
    ReturnOnError((*playerItf)->SetPlayState(playerItf, SL_PLAYSTATE_PAUSED), "Failed to set SL_PLAYSTATE_STOPPED");
  }
  return true;
}

void Sound::setVolume(int volume, int timeout_seconds) {
  if (last_volume_ == volume) {
    return;
  }
  double current_time = nanos_since_boot();
  if ((current_time - last_set_volume_time_) > (timeout_seconds * (1e+9))) {
    char volume_change_cmd[64];
    snprintf(volume_change_cmd, sizeof(volume_change_cmd), "service call audio 3 i32 3 i32 %d i32 1 &", volume);
    system(volume_change_cmd);
    last_volume_ = volume;
    last_set_volume_time_ = current_time;
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
