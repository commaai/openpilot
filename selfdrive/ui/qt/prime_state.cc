#include "selfdrive/ui/qt/prime_state.h"

#include <QJsonDocument>

#include "selfdrive/ui/qt/api.h"
#include "selfdrive/ui/qt/request_repeater.h"
#include "selfdrive/ui/qt/util.h"

PrimeState::PrimeState(QObject* parent) : QObject(parent) {
  const char *env_prime_type = std::getenv("PRIME_TYPE");
  auto type = env_prime_type ? env_prime_type : Params().get("PrimeType");

  if (!type.empty()) {
    prime_type = static_cast<PrimeState::Type>(std::atoi(type.c_str()));
  }

  if (auto dongleId = getDongleId()) {
    QString url = CommaApi::BASE_URL + "/v1.1/devices/" + *dongleId + "/";
    RequestRepeater* repeater = new RequestRepeater(this, url, "ApiCache_Device", 5);
    QObject::connect(repeater, &RequestRepeater::requestDone, this, &PrimeState::handleReply);
  }

  // Emit the initial state change
  QTimer::singleShot(1, [this]() { emit changed(prime_type); });
}

void PrimeState::handleReply(const QString& response, bool success) {
  if (!success) return;

  QJsonDocument doc = QJsonDocument::fromJson(response.toUtf8());
  if (doc.isNull()) {
    qDebug() << "JSON Parse failed on getting pairing and PrimeState status";
    return;
  }

  QJsonObject json = doc.object();
  bool is_paired = json["is_paired"].toBool();
  auto type = static_cast<PrimeState::Type>(json["prime_type"].toInt());
  setType(is_paired ? type : PrimeState::PRIME_TYPE_UNPAIRED);
}

void PrimeState::setType(PrimeState::Type type) {
  if (type != prime_type) {
    prime_type = type;
    Params().put("PrimeType", std::to_string(prime_type));
    emit changed(prime_type);
  }
}
