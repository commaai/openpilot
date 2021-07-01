#include <QApplication>
#include <QSslConfiguration>

#include "selfdrive/common/swaglog.h"
#include "selfdrive/hardware/hw.h"
#include "selfdrive/ui/qt/qt_window.h"
#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/qt/window.h"

void customMessageHandler(QtMsgType type, const QMessageLogContext &context, const QString &msg) {
  static std::map<QtMsgType, int> levels = {
    {QtMsgType::QtDebugMsg, 10},
    {QtMsgType::QtInfoMsg, 20},
    {QtMsgType::QtWarningMsg, 30},
    {QtMsgType::QtCriticalMsg, 40},
    {QtMsgType::QtSystemMsg, 40},
    {QtMsgType::QtFatalMsg, 50},
  };

  std::string file, function;
  if (context.file != nullptr) file = context.file;
  if (context.function != nullptr) function = context.function;

  auto bts = msg.toUtf8();
  cloudlog_e(levels[type], file.c_str(), context.line, function.c_str(), "%s", bts.constData());
}

int main(int argc, char *argv[]) {
  qInstallMessageHandler(customMessageHandler);
  setQtSurfaceFormat();

  if (Hardware::EON()) {
    QApplication::setAttribute(Qt::AA_ShareOpenGLContexts);
    QSslConfiguration ssl = QSslConfiguration::defaultConfiguration();
    ssl.setCaCertificates(QSslCertificate::fromPath("/usr/etc/tls/cert.pem"));
    QSslConfiguration::setDefaultConfiguration(ssl);
  }

  QApplication a(argc, argv);
  MainWindow w;
  setMainWindow(&w);
  a.installEventFilter(&w);
  return a.exec();
}
