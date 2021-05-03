#include <QUrl>
#include "sound.h"

Sound::Sound() {
  for (auto &kv : sound_map) {
    auto path = QUrl::fromLocalFile(kv.second.first);
    sounds[kv.first].setSource(path);
  }
}

void Sound::play(AudibleAlert alert) {
  int loops = sound_map[alert].second> - 1 ? sound_map[alert].second : QSoundEffect::Infinite;
  sounds[alert].setLoopCount(loops);
  sounds[alert].setVolume(volume);
  sounds[alert].play();
}

void Sound::stop() {
  for (auto &kv : sounds) {
    // Only stop repeating sounds
    if (sound_map[kv.first].second != 0) {
      kv.second.stop();
    }
  }
}
