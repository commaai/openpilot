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
  QObject::connect(window, &OnroadWindow::drewOnroadFrame, recorder, &Recorder::saveFrame, Qt::QueuedConnection);
  QObject::connect(app, &QCoreApplication::aboutToQuit, recorderThread, &QThread::quit);

  window->setAttribute(Qt::WA_DontShowOnScreen);
  window->setAttribute(Qt::WA_Mapped);
  window->setAttribute(Qt::WA_NoSystemBackground);
  recorderThread->start();

  // Initialize and start replay
  initReplay();
  replayThread = QThread::create([this] { startReplay(); });
  replayThread->start();
}

void Application::initReplay() {
  std::vector<std::string> allow;
  std::vector<std::string> block;
  replay = std::make_unique<Replay>("a2a0ccea32023010|2023-07-27--13-01-19", allow, block, nullptr,
                                  REPLAY_FLAG_NONE);
  replay->setSegmentCacheLimit(10);
}

void Application::startReplay() {
  if (!replay || !replay->load()) {
    qWarning() << "Failed to load replay";
    return;
  }

  qInfo() << "Replay started.";
  replayRunning = true;
  replay->setEndSeconds(120);
  replay->start(60);
  replay->waitUntilEnd();
  qInfo() << "Replay ended.";
  replayRunning = false;
  QMetaObject::invokeMethod(app, "quit", Qt::QueuedConnection);
}

Application::~Application() {
  if (replayThread) {
    replayThread->quit();
    replayThread->wait();
    delete replayThread;
  }

  if (recorderThread) {
    recorderThread->quit();
    recorderThread->wait();
  }

  delete window;
  delete app;
}

int Application::exec() const {
  std::this_thread::sleep_for(std::chrono::seconds(3));
  setMainWindow(window);
  app->exec();
  return 0;
}


