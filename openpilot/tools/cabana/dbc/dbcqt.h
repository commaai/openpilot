#pragma once

#include <QMetaType>
#include <QObject>

#include "tools/cabana/dbc/dbcmanager.h"

Q_DECLARE_METATYPE(MessageId)
Q_DECLARE_METATYPE(ValueDescription)

class QtDBCNotifier : public QObject {
  Q_OBJECT

public:
  explicit QtDBCNotifier(QObject *parent = nullptr);

signals:
  void signalAdded(MessageId id, const cabana::Signal *sig);
  void signalRemoved(const cabana::Signal *sig);
  void signalUpdated(const cabana::Signal *sig);
  void msgUpdated(MessageId id);
  void msgRemoved(MessageId id);
  void DBCFileChanged();
  void maskUpdated();
};

QtDBCNotifier *dbcNotifier();
