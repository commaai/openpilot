#include <stdio.h>
#include "slplay.h"

void play_sound(char *uri, int volume) {
  char **error = NULL;
  printf("call slplay_setup\n");
  slplay_setup(error);
  if (error) { printf("%s\n", *error); return; }

  printf("call slplay_create_player_for_uri\n");
  slplay_create_player_for_uri(uri, error);
  if (error) { printf("%s\n", *error); return; }

  printf("call slplay_play\n");

  while (1) {
    char volume_change_cmd[64];
    sprintf(volume_change_cmd, "service call audio 3 i32 3 i32 %d i32 1", volume);
    system(volume_change_cmd);

    slplay_play(uri, false, error);
    if (error) { printf("%s\n", *error); return; }

    sleep(1);
  }
}

int main(int argc, char *argv[]) {
  int volume = 10;
  if (argc > 2) {
    volume = atoi(argv[2]);
  }
  printf("setting volume to %d\n", volume);

  play_sound(argv[1], volume);
  return 0;
}
