#include "tools/replay/clip/application.h"

#include <QApplication>
#include <QTranslator>
#include <selfdrive/ui/qt/util.h>
#include <selfdrive/ui/qt/window.h>

#include "recorder/widget.h"

Application::Application() {

}

Application::~Application() {

}

int Application::exec(int argc, char *argv[]) {
  initApp(argc, argv);

  QApplication a(argc, argv);

  QString outputFile = "/Users/trey/Desktop/out.mp4";

  QTranslator translator;
  QString translation_file = QString::fromStdString(Params().get("LanguageSetting"));
  if (!translator.load(QString(":/%1").arg(translation_file)) && translation_file.length()) {
    qCritical() << "Failed to load translation file:" << translation_file;
  }

  a.installTranslator(&translator);

  OnroadWindow w;

  QThread recorderThread;
  Recorder recorder;
  recorder.moveToThread(&recorderThread);
  QObject::connect(&recorderThread, &QThread::finished, &recorder, &QObject::deleteLater);
  QObject::connect(&w, &OnroadWindow::drewOnroadFrame, &recorder, &Recorder::saveFrame, Qt::QueuedConnection);
  recorderThread.start();

  w.setAttribute(Qt::WA_DontShowOnScreen);
  w.setAttribute(Qt::WA_Mapped);
  w.setAttribute(Qt::WA_NoSystemBackground);
  w.resize(DEVICE_SCREEN_SIZE);
  setMainWindow(&w);

  return a.exec();
}


