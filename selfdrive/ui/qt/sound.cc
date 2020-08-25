#include <QUrl>
#include "sound.hpp"

bool Sound::init(int volume) {
  for (auto &kv : sound_map) {
    auto path = QUrl::fromLocalFile(kv.second.first);
    //sounds[kv.first] = QSoundEffect();
    sounds[kv.first].setSource(path);
  }
  return true;
}

bool Sound::play(AudibleAlert alert) {
  //sounds[alert].setLoopCount(sound_map[alert].second.second);
  sounds[alert].play();
  return true;
}

void Sound::stop() {
  sounds[alert].stop();
}

void Sound::setVolume(int volume) {
  // TODO: implement this
}

Sound::~Sound() {

}

