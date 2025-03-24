#include "tools/clip/application.h"

#include <QApplication>
#include <QTranslator>
#include <selfdrive/ui/qt/util.h>
#include <selfdrive/ui/qt/window.h>

#include "recorder/widget.h"

Application::Application(int argc, char *argv[]) {
  initApp(argc, argv);

  app = new QApplication(argc, argv);

  QString outputFile = "/Users/trey/Desktop/out.mp4";

  QTranslator translator;
  QString translation_file = QString::fromStdString(Params().get("LanguageSetting"));
  if (!translator.load(QString(":/%1").arg(translation_file)) && translation_file.length()) {
    qCritical() << "Failed to load translation file:" << translation_file;
  }

  app->installTranslator(&translator);

  window = new OnroadWindow();

  recorderThread = new QThread;
  recorder = new Recorder;
  recorder->moveToThread(recorderThread);
  QObject::connect(recorderThread, &QThread::finished, recorder, &QObject::deleteLater);
  recorderThread->start();
  QObject::connect(window, &OnroadWindow::drewOnroadFrame, recorder, &Recorder::saveFrame, Qt::QueuedConnection);

  window->setAttribute(Qt::WA_DontShowOnScreen);
  window->setAttribute(Qt::WA_Mapped);
  window->setAttribute(Qt::WA_NoSystemBackground);
}

void Application::close() const {
  recorderThread->quit();
  app->quit();
}

Application::~Application() {
  delete recorder;
  delete recorderThread;
  delete window;
  delete app;
}

int Application::exec() const {
  setMainWindow(window);
  return app->exec();
}


