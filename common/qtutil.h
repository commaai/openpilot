#pragma once

#include <QUuid>

inline QString generateUuidString() {
  return QUuid::createUuid().toString(QUuid::WithoutBraces);
}
