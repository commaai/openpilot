#include <math.h>
#include <stdlib.h>
#include <atomic>
#include "common/swaglog.h"
#include "common/timing.h"

#include "android/sl_sound.hpp"

#define LogOnError(func, msg) \
  if ((func) != SL_RESULT_SUCCESS) { LOGW(msg); }

#define ReturnOnError(func, msg) \
  if ((func) != SL_RESULT_SUCCESS) { LOGW(msg); return false; }

struct SLSound::Player {
  SLObjectItf player;
  SLPlayItf playItf;
  std::atomic<int> repeat;
};

SLSound::SLSound() {
  if (!init()){
    throw std::runtime_error("Failed to initialize sound");
  }
}

bool SLSound::init() {
  SLEngineOption engineOptions[] = {{SL_ENGINEOPTION_THREADSAFE, SL_BOOLEAN_TRUE}};
  const SLInterfaceID ids[1] = {SL_IID_VOLUME};
  const SLboolean req[1] = {SL_BOOLEAN_FALSE};
  SLEngineItf engineInterface = NULL;
  ReturnOnError(slCreateEngine(&engine_, 1, engineOptions, 0, NULL, NULL), "Failed to create OpenSL engine");
  ReturnOnError((*engine_)->Realize(engine_, SL_BOOLEAN_FALSE), "Failed to realize OpenSL engine");
  ReturnOnError((*engine_)->GetInterface(engine_, SL_IID_ENGINE, &engineInterface), "Failed to get OpenSL engine interface");
  ReturnOnError((*engineInterface)->CreateOutputMix(engineInterface, &outputMix_, 1, ids, req), "Failed to create output mix");
  ReturnOnError((*outputMix_)->Realize(outputMix_, SL_BOOLEAN_FALSE), "Failed to realize output mix");

  for (auto &kv : sound_map) {
    SLDataLocator_URI locUri = {SL_DATALOCATOR_URI, (SLchar *)kv.second.first};
    SLDataFormat_MIME formatMime = {SL_DATAFORMAT_MIME, NULL, SL_CONTAINERTYPE_UNSPECIFIED};
    SLDataSource audioSrc = {&locUri, &formatMime};
    SLDataLocator_OutputMix outMix = {SL_DATALOCATOR_OUTPUTMIX, outputMix_};
    SLDataSink audioSnk = {&outMix, NULL};

    SLObjectItf player = NULL;
    SLPlayItf playItf = NULL;
    ReturnOnError((*engineInterface)->CreateAudioPlayer(engineInterface, &player, &audioSrc, &audioSnk, 0, NULL, NULL), "Failed to create audio player");
    ReturnOnError((*player)->Realize(player, SL_BOOLEAN_FALSE), "Failed to realize audio player");
    ReturnOnError((*player)->GetInterface(player, SL_IID_PLAY, &playItf), "Failed to get player interface");
    ReturnOnError((*playItf)->SetPlayState(playItf, SL_PLAYSTATE_PAUSED), "Failed to initialize playstate to SL_PLAYSTATE_PAUSED");

    player_[kv.first] = new SLSound::Player{player, playItf};
  }
  return true;
}

void SLAPIENTRY slplay_callback(SLPlayItf playItf, void *context, SLuint32 event) {
  SLSound::Player *s = reinterpret_cast<SLSound::Player *>(context);
  if (event == SL_PLAYEVENT_HEADATEND && s->repeat != 0) {
    if (s->repeat > 0) --s->repeat;
    (*playItf)->SetPlayState(playItf, SL_PLAYSTATE_STOPPED);
    (*playItf)->SetMarkerPosition(playItf, 0);
    (*playItf)->SetPlayState(playItf, SL_PLAYSTATE_PLAYING);
  }
}

bool SLSound::play(AudibleAlert alert) {
  if (currentSound_ != AudibleAlert::NONE) {
    stop();
  }

  auto player = player_.at(alert);
  SLPlayItf playItf = player->playItf;

  int loops = sound_map[alert].second;
  player->repeat = loops > 0 ? loops - 1 : loops;
  if (player->repeat != 0) {
    ReturnOnError((*playItf)->RegisterCallback(playItf, slplay_callback, player), "Failed to register callback");
    ReturnOnError((*playItf)->SetCallbackEventsMask(playItf, SL_PLAYEVENT_HEADATEND), "Failed to set callback event mask");
  }

  // Reset the audio player
  ReturnOnError((*playItf)->ClearMarkerPosition(playItf), "Failed to clear marker position");
  uint32_t states[] = {SL_PLAYSTATE_PAUSED, SL_PLAYSTATE_STOPPED, SL_PLAYSTATE_PLAYING};
  for (auto state : states) {
    ReturnOnError((*playItf)->SetPlayState(playItf, state), "Failed to set SL_PLAYSTATE_PLAYING");
  }
  currentSound_ = alert;
  return true;
}

void SLSound::stop() {
  if (currentSound_ != AudibleAlert::NONE) {
    auto player = player_.at(currentSound_);
    player->repeat = 0;
    LogOnError((*(player->playItf))->SetPlayState(player->playItf, SL_PLAYSTATE_PAUSED), "Failed to set SL_PLAYSTATE_PAUSED");
    currentSound_ = AudibleAlert::NONE;
  }
}

void SLSound::setVolume(int volume) {
  if (last_volume_ == volume) return;

  double current_time = nanos_since_boot();
  if ((current_time - last_set_volume_time_) > (5 * (1e+9))) { // 5s timeout on updating the volume
    char volume_change_cmd[64];
    snprintf(volume_change_cmd, sizeof(volume_change_cmd), "service call audio 3 i32 3 i32 %d i32 1 &", volume);
    system(volume_change_cmd);
    last_volume_ = volume;
    last_set_volume_time_ = current_time;
  }
}

SLSound::~SLSound() {
  for (auto &kv : player_) {
    (*(kv.second->player))->Destroy(kv.second->player);
    delete kv.second;
  }
  if (outputMix_) {
    (*outputMix_)->Destroy(outputMix_);
  }
  if (engine_) {
    (*engine_)->Destroy(engine_);
  }
}
