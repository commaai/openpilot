#pragma once

#include <QObject>

class PrimeState : public QObject {
  Q_OBJECT

public:

 enum Type {
   PRIME_TYPE_UNKNOWN = -2,
   PRIME_TYPE_UNPAIRED = -1,
   PRIME_TYPE_NONE = 0,
   PRIME_TYPE_MAGENTA = 1,
   PRIME_TYPE_LITE = 2,
   PRIME_TYPE_BLUE = 3,
   PRIME_TYPE_MAGENTA_NEW = 4,
   PRIME_TYPE_PURPLE = 5,
 };

 PrimeState(QObject *parent);
 void setType(PrimeState::Type type);
 inline PrimeState::Type currentType() const { return prime_type; }
 inline bool isSubscribed() const { return prime_type > PrimeState::PRIME_TYPE_NONE; }

signals:
  void changed(PrimeState::Type prime_type);

private:
  void handleReply(const QString &response, bool success);

  PrimeState::Type prime_type = PrimeState::PRIME_TYPE_UNKNOWN;
};
