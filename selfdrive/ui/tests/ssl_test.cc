#include <QDebug>
#include <QApplication>
#include <QSslSocket>
#include <QNetworkProxy>

int main(int argc, char **argv) {
  QCoreApplication app(argc, argv);
  QSslSocket socket;

  socket.setPeerVerifyMode(QSslSocket::VerifyNone);

  qDebug() << "Supports SSL: " << QSslSocket::supportsSsl();
  qDebug() << "Version: " << QSslSocket::sslLibraryVersionString() << " " << QSslSocket::sslLibraryVersionNumber();
  qDebug() << "Build Version: " << QSslSocket::sslLibraryBuildVersionString() << " " << QSslSocket::sslLibraryBuildVersionNumber();

  socket.connectToHost("www.google.com", 443);

  socket.write("GET / HTTP/1.1rnrn");

  while (socket.waitForReadyRead())
    qDebug() << socket.readAll().data();
}
