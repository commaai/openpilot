#include "selfdrive/ui/qt/api.h"

#include <QApplication>
#include <QDebug>
#include <QNetworkRequest>

#include "selfdrive/ui/qt/util.h"

HttpRequest::HttpRequest(QObject *parent, bool create_jwt, int timeout) : create_jwt(create_jwt), QObject(parent) {
  networkTimer = new QTimer(this);
  networkTimer->setSingleShot(true);
  networkTimer->setInterval(timeout);
  connect(networkTimer, &QTimer::timeout, this, &HttpRequest::requestTimeout);
}

bool HttpRequest::active() const {
  return reply != nullptr;
}

bool HttpRequest::timeout() const {
  return reply && reply->error() == QNetworkReply::OperationCanceledError;
}

void HttpRequest::sendRequest(const QString &requestURL, const HttpRequest::Method method) {
  if (active()) {
    qDebug() << "HttpRequest is active";
    return;
  }
  QString token = QString::fromStdString(CommaApi::create_token(create_jwt));
  QNetworkRequest request;
  request.setUrl(QUrl(requestURL));
  request.setRawHeader("User-Agent", getUserAgent().toUtf8());

  if (!token.isEmpty()) {
    request.setRawHeader(QByteArray("Authorization"), ("JWT " + token).toUtf8());
  }

  if (method == HttpRequest::Method::GET) {
    reply = nam()->get(request);
  } else if (method == HttpRequest::Method::DELETE) {
    reply = nam()->deleteResource(request);
  }

  networkTimer->start();
  connect(reply, &QNetworkReply::finished, this, &HttpRequest::requestFinished);
}

void HttpRequest::requestTimeout() {
  reply->abort();
}

void HttpRequest::requestFinished() {
  networkTimer->stop();

  if (reply->error() == QNetworkReply::NoError) {
    emit requestDone(reply->readAll(), true, reply->error());
  } else {
    QString error;
    if (reply->error() == QNetworkReply::OperationCanceledError) {
      nam()->clearAccessCache();
      nam()->clearConnectionCache();
      error = "Request timed out";
    } else {
      error = reply->errorString();
    }
    emit requestDone(error, false, reply->error());
  }

  reply->deleteLater();
  reply = nullptr;
}

QNetworkAccessManager *HttpRequest::nam() {
  static QNetworkAccessManager *networkAccessManager = new QNetworkAccessManager(qApp);
  return networkAccessManager;
}
