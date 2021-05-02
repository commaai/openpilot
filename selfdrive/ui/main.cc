#include <QApplication>
#include <QSslConfiguration>

#include "qt/window.hpp"
#include "qt/qt_window.hpp"

int main(int argc, char *argv[]) {
  QSurfaceFormat fmt;
#ifdef __APPLE__
  fmt.setVersion(3, 2);
  fmt.setProfile(QSurfaceFormat::OpenGLContextProfile::CoreProfile);
  fmt.setRenderableType(QSurfaceFormat::OpenGL);
#else
  fmt.setRenderableType(QSurfaceFormat::OpenGLES);
#endif
  QSurfaceFormat::setDefaultFormat(fmt);

#ifdef QCOM
  QApplication::setAttribute(Qt::AA_ShareOpenGLContexts);
#endif

#ifdef QCOM
  QSslConfiguration ssl = QSslConfiguration::defaultConfiguration();
  ssl.setCaCertificates(QSslCertificate::fromPath("/usr/etc/tls/cert.pem", QSsl::Pem, QRegExp::Wildcard));
  QSslConfiguration::setDefaultConfiguration(ssl);
#endif

  QApplication a(argc, argv);
  MainWindow w;
  setMainWindow(&w);
  a.installEventFilter(&w);
  return a.exec();
}
