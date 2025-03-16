/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#include "selfdrive/ui/sunnypilot/qt/api.h"

#include <QApplication>
#include <QJsonDocument>

#include "util.h"
#include "selfdrive/ui/qt/util.h"

namespace SunnylinkApi {
  QString create_jwt(const QJsonObject &payloads, int expiry, bool sunnylink) {
    QJsonObject header = {{"alg", "RS256"}};

    auto t = QDateTime::currentSecsSinceEpoch();
    auto dongle_id = sunnylink ? getSunnylinkDongleId() : getDongleId();
    QJsonObject payload = {{"identity", dongle_id.value_or("")}, {"nbf", t}, {"iat", t}, {"exp", t + expiry}};
    for (auto it = payloads.begin(); it != payloads.end(); ++it) {
      payload.insert(it.key(), it.value());
    }

    auto b64_opts = QByteArray::Base64UrlEncoding | QByteArray::OmitTrailingEquals;
    QString jwt = QJsonDocument(header).toJson(QJsonDocument::Compact).toBase64(b64_opts) + '.' +
                  QJsonDocument(payload).toJson(QJsonDocument::Compact).toBase64(b64_opts);

    auto hash = QCryptographicHash::hash(jwt.toUtf8(), QCryptographicHash::Sha256);
    return jwt + "." + CommaApi::rsa_sign(hash).toBase64(b64_opts);
  }
} // namespace SunnylinkApi

void HttpRequestSP::sendRequest(const QString& requestURL, Method method, const QByteArray& payload) {
  if (active()) {
    return;
  }
  QNetworkRequest request = prepareRequest(requestURL);

  if(!payload.isEmpty() && (method == Method::POST || method == Method::PUT)) {
    request.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");
  }

  switch (method) {
    case Method::GET:
      reply = nam()->get(request);
      break;
    case Method::DELETE:
      reply = nam()->deleteResource(request);
      break;
    case Method::POST:
      reply = nam()->post(request, payload);
      break;
    case Method::PUT:
      reply = nam()->put(request, payload);
      break;
  }

  networkTimer->start();
  connect(reply, &QNetworkReply::finished, this, &HttpRequestSP::requestFinished);
}
