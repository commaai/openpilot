#include <QUrl>
#include "sound.hpp"

Sound::Sound() {
  for (auto &kv : sound_map) {
    auto path = QUrl::fromLocalFile(kv.second.first);
    sounds[kv.first].setSource(path);
  }
}

bool Sound::play(AudibleAlert alert) {
  sounds[alert].setLoopCount(sound_map[alert].second);
  sounds[alert].play();
  return true;
}

void Sound::stop() {
  for (auto &kv : sounds) {
    kv.second.stop();
  }
}

void Sound::setVolume(int volume) {
  // TODO: implement this
}

Sound::~Sound() {

}

