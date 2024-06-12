#include <QApplication>
#include <QSoundEffect>
#include <QTimer>
#include <QDebug>

int main(int argc, char **argv) {

  QApplication a(argc, argv);

  QTimer::singleShot(0, [=]{
    QSoundEffect s;
    const char *vol = getenv("VOLUME");
    s.setVolume(vol ? atof(vol) : 1.0);
    for (int i = 1; i < argc; i++) {
      QString fn = argv[i];
      qDebug() << "playing" << fn;

      QEventLoop loop;
      s.setSource(QUrl::fromLocalFile(fn));
      QEventLoop::connect(&s, &QSoundEffect::loadedChanged, &loop, &QEventLoop::quit);
      loop.exec();
      s.play();
      QEventLoop::connect(&s, &QSoundEffect::playingChanged, &loop, &QEventLoop::quit);
      loop.exec();
    }
    QCoreApplication::exit();
  });

  return a.exec();
}
