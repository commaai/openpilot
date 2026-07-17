#include "tools/cabana/dbc/dbcqt.h"

QtDBCNotifier::QtDBCNotifier(QObject *parent) : QObject(parent) {
  dbc()->setCallbacks({
    .signal_added = [this](MessageId id, const cabana::Signal *sig) { emit signalAdded(id, sig); },
    .signal_removed = [this](const cabana::Signal *sig) { emit signalRemoved(sig); },
    .signal_updated = [this](const cabana::Signal *sig) { emit signalUpdated(sig); },
    .msg_updated = [this](MessageId id) { emit msgUpdated(id); },
    .msg_removed = [this](MessageId id) { emit msgRemoved(id); },
    .file_changed = [this]() { emit DBCFileChanged(); },
    .mask_updated = [this]() { emit maskUpdated(); },
  });
}

QtDBCNotifier *dbcNotifier() {
  static QtDBCNotifier notifier;
  return &notifier;
}
