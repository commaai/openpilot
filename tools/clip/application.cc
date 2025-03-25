#include "tools/clip/application.h"

#include <QApplication>
#include <QTranslator>
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

  parser.addPositionalArgument("route", "route string");

  parser.process(*app);

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
  QObject::connect(recorderThread, &QThread::finished, recorder, &QObject::deleteLater);

  QObject::connect(window, &OnroadWindow::redrew, this, [&]() {
    QElapsedTimer timer;
    timer.start();
    QPixmap pixmap = window->grab();
    // qDebug() << "pixmap took " << timer.elapsed() << " ms";
    timer.restart();
    recorder->saveFrame(std::make_shared<QPixmap>(std::move(pixmap)));
    // qDebug() << "save frame took" << timer.elapsed() << " ms";
  }, Qt::QueuedConnection);

  QObject::connect(app, &QCoreApplication::aboutToQuit, recorderThread, &QThread::quit);

  // window->setAttribute(Qt::WA_DontShowOnScreen);
  window->setAttribute(Qt::WA_Mapped);
  window->setAttribute(Qt::WA_NoSystemBackground);
  recorderThread->start();

  // Initialize and start replay
  initReplay(route.toStdString());
  replayThread = QThread::create([this, startTime] { startReplay(startTime); });
  replayThread->start();
}

void Application::initReplay(const std::string& route) {
  std::vector<std::string> allow;
  std::vector<std::string> block;
  replay = std::make_unique<Replay>(route, allow, block, nullptr, REPLAY_FLAG_NONE);
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
  }

  delete window;
  delete app;
}

int Application::exec() const {
  // TODO: modify Replay to block until all OnroadWindow required messages have been broadcast at least once
  std::this_thread::sleep_for(std::chrono::seconds(5));
  setMainWindow(window);
  return app->exec();
}


