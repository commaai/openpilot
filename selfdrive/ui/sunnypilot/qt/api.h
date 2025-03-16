/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#pragma once

#include "selfdrive/ui/qt/api.h"
#include "selfdrive/ui/sunnypilot/qt/util.h"
#include "common/util.h"

namespace SunnylinkApi {
  QByteArray rsa_encrypt(const QByteArray& data);
  QByteArray rsa_decrypt(const QByteArray& data);
  QString create_jwt(const QJsonObject& payloads = {}, int expiry = 3600, bool sunnylink = false);
}

class HttpRequestSP : public HttpRequest {
  Q_OBJECT

public:
  explicit HttpRequestSP(QObject* parent, bool create_jwt = true, int timeout = 20000, bool sunnylink = false) :
    HttpRequest(parent, create_jwt, timeout), sunnylink(sunnylink) {}

  using HttpRequest::sendRequest;
  void sendRequest(const QString& requestURL, Method method, const QByteArray& payload);

private:
  bool sunnylink;

protected:
  [[nodiscard]] QString GetJwtToken() const override { return SunnylinkApi::create_jwt({}, 3600, sunnylink); }
  [[nodiscard]] QString GetUserAgent() const override { return getUserAgent(sunnylink); }
};
