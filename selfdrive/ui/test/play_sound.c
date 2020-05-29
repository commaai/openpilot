#include <stdio.h>
#include "sound.hpp"

void play_sound(AudibleAlert alert, int volume) {
  Sound sound;
  assert(sound.init(volumn));
  while (1) {
    sound.play(alert);
    sleep(1);
  }
}

int main(int argc, char *argv[]) {
  int volume = 10;
  if (argc > 2) {
    volume = atoi(argv[2]);
  }
  printf("setting volume to %d\n", volume);
  play_sound((AudibleAlert)(atoi(argv[1])), volume);
  return 0;
}
