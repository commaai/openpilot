#include <stdlib.h>
#include "sound.hpp"

int main(int arg, char* argv[]) {

  int sound = atoi(argv[1]);
  int volume = atoi(argv[2]);

  Sound sound;
  sound.init(volume);
  sound.play(sound);

  return 0;
}
