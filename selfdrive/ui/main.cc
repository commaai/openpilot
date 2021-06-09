#include <QApplication>
#include <QSslConfiguration>

#include "selfdrive/hardware/hw.h"
#include "selfdrive/ui/qt/qt_window.h"
#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/qt/window.h"

int main(int argc, char *argv[]) {
  qInstallMessageHandler(swagLogMessageHandler);
  initApp();

  if (Hardware::EON()) {
    QSslConfiguration ssl = QSslConfiguration::defaultConfiguration();
    ssl.setCaCertificates(QSslCertificate::fromPath("/usr/etc/tls/cert.pem"));
    QSslConfiguration::setDefaultConfiguration(ssl);
  }

  QApplication a(argc, argv);

//  QLocale curLocale(QLocale("fr_FR"));
//  QLocale::setDefault(curLocale);
  QTranslator translator;
  if (!translator.load("main_fr", "translations")) {
    qDebug() << "Failed to load translation!";
  }
  a.installTranslator(&translator);  // needs to be before setting main window

  MainWindow w;
  setMainWindow(&w);
  a.installEventFilter(&w);
  return a.exec();
}
