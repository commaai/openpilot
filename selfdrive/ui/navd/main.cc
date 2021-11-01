#include <termios.h>

#include <QApplication>
#include <QCommandLineParser>
#include <QDebug>
#include <QThread>
#include <csignal>
#include <iostream>

void sigHandler(int s) {
  qInfo() << "Shutting down";
  std::signal(s, SIG_DFL);
  qApp->quit();
}


int main(int argc, char *argv[]) {
  QApplication app(argc, argv);
  std::signal(SIGINT, sigHandler);
  std::signal(SIGTERM, sigHandler);

   QCommandLineParser parser;
  parser.setApplicationDescription("Navigation server. Runs stand-alone, or using pre-computer route");
  parser.addHelpOption();
  parser.process(app);
  const QStringList args = parser.positionalArguments();

  return app.exec();
}
