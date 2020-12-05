#include <QUrl>
#include "qt_sound.hpp"

QtSound::QtSound() {
  for (int i = 0; i < sizeof(sound_map) / sizeof(sound_map[0]); ++i) {
    auto [file_path, loop_count] = sound_map[i];
    if (!file_path) continue;

    auto path = QUrl::fromLocalFile(file_path);
    sounds[(AudibleAlert)i].setSource(path);
  }
}

bool QtSound::play(AudibleAlert alert) {
  const int loops = sound_map[(int)alert].second;
  sounds[alert].setLoopCount(loops > - 1 ? loops : QSoundEffect::Infinite);
  sounds[alert].setVolume(0.7);
  sounds[alert].play();
  return true;
}

void QtSound::stop() {
  for (auto &kv : sounds) {
    // Only stop repeating sounds
    if (sound_map[(int)kv.first].second != 0) {
      kv.second.stop();
    }
  }
}

void QtSound::setVolume(int volume) {
  // TODO: implement this
}
