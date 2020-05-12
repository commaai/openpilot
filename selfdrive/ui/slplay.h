#ifndef SLPLAY_H
#define SLPLAY_H

#include <SLES/OpenSLES.h>
#include <SLES/OpenSLES_Android.h>
#include <stdbool.h>

typedef struct {
  const char* uri;
  SLObjectItf player;
  SLPlayItf playInterface;
} uri_player;

void slplay_setup(char **error);
uri_player* slplay_create_player_for_uri(const char* uri, char **error);
void slplay_play (const char *uri, bool loop, char **error);
void slplay_stop_uri (const char* uri, char **error);
void slplay_destroy();

#endif

