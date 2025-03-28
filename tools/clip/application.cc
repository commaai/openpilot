#include "tools/clip/application.h"

#include <QApplication>
#include <QTranslator>
#include <QWindow>
#include <selfdrive/ui/qt/util.h>
#include <selfdrive/ui/qt/window.h>

#include "recorder/widget.h"

Application::Application(int argc, char *argv[], QObject *parent) : QObject(parent) {
  argc_ = argc;
  argv_ = argv;

  initApp(argc_, argv_);

  app = new QApplication(argc_, argv_);

  QCommandLineParser parser;
  parser.setApplicationDescription("Clip your ride!");
  parser.addHelpOption();

  const QCommandLineOption start({"s", "start"}, "start time", "start");
  parser.addOption(start);

  const QCommandLineOption output({"o", "output"}, "output file", "output");
  parser.addOption(output);

  const QCommandLineOption data_dir_arg({"d", "data_dir"}, "data directory", "data_dir");
  parser.addOption(data_dir_arg);

  parser.addPositionalArgument("route", "route string");

  parser.process(*app);

  const QString data_dir = parser.value(data_dir_arg);

  int startTime = 0;
  if (parser.isSet(start)) {
    bool ok;
    int parsed = parser.value(start).toInt(&ok);
    if (!ok) {
      qDebug() << "start time must be an integer\n";
      fprintf(stderr, "%s", parser.helpText().toStdString().c_str());
      exit(1);
    }
    startTime = parsed;
  }

  if (!parser.isSet(output)) {
    qDebug() << "output is required\n";
    fprintf(stderr, "%s", parser.helpText().toStdString().c_str());
    exit(1);
  }
  QString outputFile = parser.value(output);

  QString route;
  QStringList positionalArgs = parser.positionalArguments();
  if (!positionalArgs.isEmpty()) {
    route = positionalArgs.at(0);
  } else {
    qDebug() << "No file specified\n";
    fprintf(stderr, "%s", parser.helpText().toStdString().c_str());
    exit(1);
  }

  QTranslator translator;
  QString translation_file = QString::fromStdString(Params().get("LanguageSetting"));
  if (!translator.load(QString(":/%1").arg(translation_file)) && translation_file.length()) {
    qCritical() << "Failed to load translation file:" << translation_file;
  }

  app->installTranslator(&translator);

  window = new OnroadWindow();

  recorderThread = new QThread;
  recorder = new Recorder(outputFile.toStdString());
  recorder->moveToThread(recorderThread);
  connect(recorderThread, &QThread::finished, recorder, &QObject::deleteLater);
  connect(app, &QCoreApplication::aboutToQuit, recorderThread, &QThread::quit);
  recorderThread->start();

  // Initialize and start replay
  initReplay(route.toStdString(), data_dir.isEmpty() ? "" : data_dir.toStdString());
  replayThread = QThread::create([this, startTime] { startReplay(startTime); });
  replayThread->start();

  // Frame capture optimization
  QElapsedTimer frameTimer;
  frameTimer.start();
  int64_t lastFrameTime = 0;
  const int64_t frameInterval = 1000 / UI_FREQ;  // Target frame interval in ms

  loop = new QTimer;
  connect(loop, &QTimer::timeout, this, [&, frameTimer, lastFrameTime]() mutable {
    if (!window->isVisible()) {
      return;
    }

    int64_t currentTime = frameTimer.elapsed();
    int64_t elapsedSinceLastFrame = currentTime - lastFrameTime;

    // Skip frame if we're ahead of schedule
    if (elapsedSinceLastFrame < frameInterval) {
      return;
    }

    QPixmap pixmap = window->grab();

    // Only process frame if capture was successful
    if (!pixmap.isNull()) {
      recorder->saveFrame(std::make_shared<QPixmap>(std::move(pixmap)));
      lastFrameTime = currentTime;
    }
  });

  // Use a higher timer resolution for more precise frame timing
  loop->setTimerType(Qt::PreciseTimer);
  loop->start(1);  // Run at highest possible frequency, we'll control frame rate ourselves

  window->setAttribute(Qt::WA_DontShowOnScreen);
  window->setAttribute(Qt::WA_OpaquePaintEvent);
  window->setAttribute(Qt::WA_NoSystemBackground);
  window->setAttribute(Qt::WA_TranslucentBackground, false);
  window->setAttribute(Qt::WA_AlwaysStackOnTop);
  window->setAttribute(Qt::WA_ShowWithoutActivating);
  window->setAttribute(Qt::WA_UpdatesDisabled);
  window->setAttribute(Qt::WA_StaticContents);
}

void Application::initReplay(const std::string& route, const std::string& data_dir) {
  std::vector<std::string> allow;
  std::vector<std::string> block;
  replay = std::make_unique<Replay>(route, allow, block, nullptr, REPLAY_FLAG_NONE, data_dir);
  replay->setSegmentCacheLimit(1);
}

void Application::startReplay(int start) {
  if (!replay || !replay->load()) {
    qWarning() << "Failed to load replay";
    QApplication::instance()->quit();
  }

  qInfo() << "Replay started.";
  replayRunning = true;
  replay->setEndSeconds(start + 60);
  replay->start(start + 2);
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
    delete recorderThread;
  }

  delete recorder;
  delete window;
  delete loop;
  delete app;
}

int Application::exec() const {
  // TODO: modify Replay to block until all OnroadWindow required messages have been broadcast at least once
  std::this_thread::sleep_for(std::chrono::seconds(8));
  setMainWindow(window);
  return app->exec();
}


