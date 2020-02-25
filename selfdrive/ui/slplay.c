#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <stdlib.h>
#include <getopt.h>

#include "common/timing.h"
#include "slplay.h"

SLEngineItf engineInterface = NULL;
SLObjectItf outputMix = NULL;
SLObjectItf engine = NULL;
uri_player players[32] = {{NULL, NULL, NULL}};

uint64_t loop_start = 0;
uint64_t loop_start_ctx = 0;

uri_player* get_player_by_uri(const char* uri) {
  for (uri_player *s = players; s->uri != NULL; s++) {
    if (strcmp(s->uri, uri) == 0) {
      return s;
    }
  }

  return NULL;
}

uri_player* slplay_create_player_for_uri(const char* uri, char **error) {
  uri_player player = { uri, NULL, NULL };

  SLresult result;
  SLDataLocator_URI locUri = {SL_DATALOCATOR_URI, (SLchar *) uri};
  SLDataFormat_MIME formatMime = {SL_DATAFORMAT_MIME, NULL, SL_CONTAINERTYPE_UNSPECIFIED};
  SLDataSource audioSrc = {&locUri, &formatMime};

  SLDataLocator_OutputMix outMix = {SL_DATALOCATOR_OUTPUTMIX, outputMix};
  SLDataSink audioSnk = {&outMix, NULL};

  result = (*engineInterface)->CreateAudioPlayer(engineInterface, &player.player, &audioSrc, &audioSnk, 0, NULL, NULL);
  if (result != SL_RESULT_SUCCESS) {
    *error = "Failed to create audio player";
    return NULL;
  }

  result = (*(player.player))->Realize(player.player, SL_BOOLEAN_FALSE);
  if (result != SL_RESULT_SUCCESS) {
    *error = "Failed to realize audio player";
    return NULL;
  }

  result = (*(player.player))->GetInterface(player.player, SL_IID_PLAY, &(player.playInterface));
  if (result != SL_RESULT_SUCCESS) {
    *error = "Failed to get player interface";
    return NULL;
  }

  result = (*(player.playInterface))->SetPlayState(player.playInterface, SL_PLAYSTATE_PAUSED);
  if (result != SL_RESULT_SUCCESS) {
    *error = "Failed to initialize playstate to SL_PLAYSTATE_PAUSED";
    return NULL;
  }

  uri_player *p = players;
  while (p->uri != NULL) {
    p++;
  }
  *p = player;

  return p;
}

void slplay_setup(char **error) {
  SLresult result;
  SLEngineOption engineOptions[] = {{SL_ENGINEOPTION_THREADSAFE, SL_BOOLEAN_TRUE}};
  result = slCreateEngine(&engine, 1, engineOptions, 0, NULL, NULL);
  if (result != SL_RESULT_SUCCESS) {
    *error = "Failed to create OpenSL engine";
  }

  result = (*engine)->Realize(engine, SL_BOOLEAN_FALSE);
  if (result != SL_RESULT_SUCCESS) {
    *error = "Failed to realize OpenSL engine";
  }

  result = (*engine)->GetInterface(engine, SL_IID_ENGINE, &engineInterface);
  if (result != SL_RESULT_SUCCESS) {
    *error = "Failed to realize OpenSL engine";
  }

  const SLInterfaceID ids[1] = {SL_IID_VOLUME};
  const SLboolean req[1] = {SL_BOOLEAN_FALSE};
  result = (*engineInterface)->CreateOutputMix(engineInterface, &outputMix, 1, ids, req);
  if (result != SL_RESULT_SUCCESS) {
    *error = "Failed to create output mix";
  }

  result = (*outputMix)->Realize(outputMix, SL_BOOLEAN_FALSE);
  if (result != SL_RESULT_SUCCESS) {
    *error = "Failed to realize output mix";
  }
}

void slplay_destroy() {
  for (uri_player *player = players; player->uri != NULL; player++) {
    if (player->player) {
      (*(player->player))->Destroy(player->player);
    }
  }

  (*outputMix)->Destroy(outputMix);
  (*engine)->Destroy(engine);
}

void slplay_stop(uri_player* player, char **error) {
  SLPlayItf playInterface = player->playInterface;
  SLresult result;

  // stop a loop
  loop_start = 0;

  result = (*playInterface)->SetPlayState(playInterface, SL_PLAYSTATE_PAUSED);
  if (result != SL_RESULT_SUCCESS) {
    *error = "Failed to set SL_PLAYSTATE_STOPPED";
    return;
  }
}

void slplay_stop_uri(const char* uri, char **error) {
  uri_player* player = get_player_by_uri(uri);
  slplay_stop(player, error);
}

void SLAPIENTRY slplay_callback(SLPlayItf playItf, void *context, SLuint32 event) {
  uint64_t cb_loop_start = *((uint64_t*)context);
  if (event == SL_PLAYEVENT_HEADATEND && cb_loop_start == loop_start) {
    (*playItf)->SetPlayState(playItf, SL_PLAYSTATE_STOPPED);
    (*playItf)->SetMarkerPosition(playItf, 0);
    (*playItf)->SetPlayState(playItf, SL_PLAYSTATE_PLAYING);
  }
}

void slplay_play (const char *uri, bool loop, char **error) {
  SLresult result;

  uri_player* player = get_player_by_uri(uri);
  if (player == NULL) {
    player = slplay_create_player_for_uri(uri, error);
    if (*error) {
      return;
    }
  }

  SLPlayItf playInterface = player->playInterface;
  if (loop) {
    loop_start = nanos_since_boot();
    loop_start_ctx = loop_start;
    result = (*playInterface)->RegisterCallback(playInterface, slplay_callback, &loop_start_ctx);
    if (result != SL_RESULT_SUCCESS) {
      char error[64];
      snprintf(error, sizeof(error), "Failed to register callback. %d", result);
      *error = error[0];
      return;
    }

    result = (*playInterface)->SetCallbackEventsMask(playInterface, SL_PLAYEVENT_HEADATEND);
    if (result != SL_RESULT_SUCCESS) {
      *error = "Failed to set callback event mask";
      return;
    }
  }

  // Reset the audio player
  result = (*playInterface)->ClearMarkerPosition(playInterface);
  if (result != SL_RESULT_SUCCESS) {
    *error = "Failed to clear marker position";
    return;
  }
  result = (*playInterface)->SetPlayState(playInterface, SL_PLAYSTATE_PAUSED);
  result = (*playInterface)->SetPlayState(playInterface, SL_PLAYSTATE_STOPPED);
  result = (*playInterface)->SetPlayState(playInterface, SL_PLAYSTATE_PLAYING);
  if (result != SL_RESULT_SUCCESS) {
    *error = "Failed to set SL_PLAYSTATE_PLAYING";
  }
}
