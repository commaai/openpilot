#include "tools/cabana/dbcmanager.h"
#include <QDebug>

#include <QFile>
#include <QRegularExpression>
#include <QTextStream>
#include <QVector>
#include <limits>
#include <sstream>


bool DBCManager::open(SourceSet s, const QString &dbc_file_name, QString *error) {
  // TODO: check if file is already open and merge sources

  try {
    dbc_files.push_back({s, new DBCFile(dbc_file_name, this)});
  } catch (std::exception &e) {
    if (error) *error = e.what();
    return false;
  }

  emit DBCFileChanged();
  return true;
}

bool DBCManager::open(SourceSet s, const QString &name, const QString &content, QString *error) {
  // TODO: check if file is already open and merge sources

  try {
    dbc_files.push_back({s, new DBCFile(name, content, this)});
  } catch (std::exception &e) {
    if (error) *error = e.what();
    return false;
  }

  emit DBCFileChanged();
  return true;
}

QString DBCManager::generateDBC() {
  // TODO: move saving logic into DBCManager
  return "";
}


void DBCManager::addSignal(const MessageId &id, const cabana::Signal &sig) {
  DBCFile *dbc_file = findDBCFile(id);
  assert(dbc_file != nullptr);
  cabana::Signal *s = dbc_file->addSignal(id, sig);

  if (s != nullptr) {
    // This DBC applies to all active sources, emit for every source
    for (uint8_t source : sources) {
      emit signalAdded({.source = source, .address = id.address}, s);
    }
  }
}

void DBCManager::updateSignal(const MessageId &id, const QString &sig_name, const cabana::Signal &sig) {
  DBCFile *dbc_file = findDBCFile(id);
  assert(dbc_file != nullptr);
  cabana::Signal *s = dbc_file->updateSignal(id, sig_name, sig);

  if (s != nullptr) {
    emit signalUpdated(s);
  }
}

void DBCManager::removeSignal(const MessageId &id, const QString &sig_name) {
  DBCFile *dbc_file = findDBCFile(id);
  assert(dbc_file != nullptr);
  cabana::Signal *s = dbc_file->removeSignal(id, sig_name);

  if (s != nullptr) {
    emit signalRemoved(s);
  }
}

void DBCManager::updateMsg(const MessageId &id, const QString &name, uint32_t size) {
  DBCFile *dbc_file = findDBCFile(id);
  assert(dbc_file != nullptr);
  dbc_file->updateMsg(id, name, size);

  // This DBC applies to all active sources, emit for every source
  for (uint8_t source : sources) {
    emit msgUpdated({.source = source, .address = id.address});
  }
}

void DBCManager::removeMsg(const MessageId &id) {
  DBCFile *dbc_file = findDBCFile(id);
  assert(dbc_file != nullptr);
  dbc_file->removeMsg(id);

  // This DBC applies to all active sources, emit for every source
  for (uint8_t source : sources) {
    emit msgRemoved({.source = source, .address = id.address});
  }
}

std::map<MessageId, cabana::Msg> DBCManager::getMessages(uint8_t source) {
  std::map<MessageId, cabana::Msg> ret;

  DBCFile *dbc_file = findDBCFile({.source = source, .address = 0});
  assert(dbc_file != nullptr);

  for (auto &[address, msg] : dbc_file->getMessages()) {
    MessageId id = {.source = source, .address = address};
    ret[id] = msg;
  }
  return ret;
}

const cabana::Msg *DBCManager::msg(const MessageId &id) const {
  DBCFile *dbc_file = findDBCFile(id);
  assert(dbc_file != nullptr);
  return dbc_file->msg(id);
}

const cabana::Msg* DBCManager::msg(uint8_t source, const QString &name) {
  DBCFile *dbc_file = findDBCFile({.source = source, .address = 0});
  assert(dbc_file != nullptr);
  return dbc_file->msg(name);
}

QStringList DBCManager::signalNames() const {
  QStringList ret;

  for (auto &[_, dbc_file] : dbc_files) {
    ret << dbc_file->signalNames();
  }

  ret.sort();
  ret.removeDuplicates();
  return ret;
}

int DBCManager::msgCount() const {
  int ret = 0;

  for (auto &[_, dbc_file] : dbc_files) {
    ret += dbc_file->msgCount();
  }

  return ret;
}

void DBCManager::updateSources(const QSet<uint8_t> &s) {
  sources = s;
}

DBCFile *DBCManager::findDBCFile(const MessageId &id) const {
  // Find DBC file that matches id.source, fall back to SOURCE_ALL if no specific DBC is found
  for (auto &[source_set, dbc_file] : dbc_files) {
    if (source_set.contains(id.source)) return dbc_file;
  }
  for (auto &[source_set, dbc_file] : dbc_files) {
    if (source_set == SOURCE_ALL) return dbc_file;
  }
  return nullptr;

}

DBCManager *dbc() {
  static DBCManager dbc_manager(nullptr);
  return &dbc_manager;
}
