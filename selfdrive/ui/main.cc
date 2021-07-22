#include <sys/resource.h>

#include <QApplication>
#include <QSslConfiguration>

#include "selfdrive/hardware/hw.h"
#include "selfdrive/ui/qt/qt_window.h"
#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/qt/window.h"

int main(int argc, char *argv[]) {
  setpriority(PRIO_PROCESS, 0, -20);

  qInstallMessageHandler(swagLogMessageHandler);
  initApp();

  if (Hardware::EON()) {
    QSslConfiguration ssl = QSslConfiguration::defaultConfiguration();
    ssl.setCaCertificates(QSslCertificate::fromPath("/usr/etc/tls/cert.pem"));
    QSslConfiguration::setDefaultConfiguration(ssl);
  }

  QApplication a(argc, argv);
  MainWindow w;
  QList<QPushButton*> pushButtons = w.findChildren<QPushButton *>();
  for (int i = 0; i < pushButtons.size(); i++) {
    pushButtons.at(i)->setAttribute(Qt::WA_AcceptTouchEvents);
    pushButtons.at(i)->setFocusPolicy(Qt::NoFocus);
  }
  qDebug() << "Set up" << pushButtons.size() << "buttons";

  setMainWindow(&w);
  a.installEventFilter(&w);
  a.setAttribute(Qt::AA_SynthesizeMouseForUnhandledTouchEvents, false);
  w.setAttribute(Qt::WA_AcceptTouchEvents);

  return a.exec();
}
