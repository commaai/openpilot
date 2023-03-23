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

void DBCManager::closeAll() {
  while (dbc_files.size()) {
    DBCFile *dbc_file = dbc_files.back().second;
    dbc_files.pop_back();
    delete dbc_file;
  }
  emit DBCFileChanged();
}


QString DBCManager::generateDBC() {
  // TODO: move saving logic into DBCManager
  return "";
}


void DBCManager::addSignal(const MessageId &id, const cabana::Signal &sig) {
  auto sources_dbc_file = findDBCFile(id);
  assert(sources_dbc_file); // Create new DBC?
  auto [dbc_sources, dbc_file] = *sources_dbc_file;

  cabana::Signal *s = dbc_file->addSignal(id, sig);

  if (s != nullptr) {
    for (uint8_t source : dbc_sources) {
      emit signalAdded({.source = source, .address = id.address}, s);
    }
  }
}

void DBCManager::updateSignal(const MessageId &id, const QString &sig_name, const cabana::Signal &sig) {
  auto sources_dbc_file = findDBCFile(id);
  assert(sources_dbc_file); // This should be impossible
  auto [_, dbc_file] = *sources_dbc_file;

  cabana::Signal *s = dbc_file->updateSignal(id, sig_name, sig);

  if (s != nullptr) {
    emit signalUpdated(s);
  }
}

void DBCManager::removeSignal(const MessageId &id, const QString &sig_name) {
  auto sources_dbc_file = findDBCFile(id);
  assert(sources_dbc_file); // This should be impossible
  auto [_, dbc_file] = *sources_dbc_file;

  cabana::Signal *s = dbc_file->removeSignal(id, sig_name);

  if (s != nullptr) {
    emit signalRemoved(s);
  }
}

void DBCManager::updateMsg(const MessageId &id, const QString &name, uint32_t size) {
  auto sources_dbc_file = findDBCFile(id);
  assert(sources_dbc_file); // This should be impossible
  auto [dbc_sources, dbc_file] = *sources_dbc_file;

  dbc_file->updateMsg(id, name, size);

  for (uint8_t source : dbc_sources) {
    emit msgUpdated({.source = source, .address = id.address});
  }
}

void DBCManager::removeMsg(const MessageId &id) {
  auto sources_dbc_file = findDBCFile(id);
  assert(sources_dbc_file); // This should be impossible
  auto [dbc_sources, dbc_file] = *sources_dbc_file;

  dbc_file->removeMsg(id);

  for (uint8_t source : dbc_sources) {
    emit msgRemoved({.source = source, .address = id.address});
  }
}

std::map<MessageId, cabana::Msg> DBCManager::getMessages(uint8_t source) {
  std::map<MessageId, cabana::Msg> ret;

  auto sources_dbc_file = findDBCFile({.source = source, .address = 0});
  if (!sources_dbc_file) {
    return ret;
  }

  auto [_, dbc_file] = *sources_dbc_file;

  for (auto &[address, msg] : dbc_file->getMessages()) {
    MessageId id = {.source = source, .address = address};
    ret[id] = msg;
  }
  return ret;
}

const cabana::Msg *DBCManager::msg(const MessageId &id) const {
  auto sources_dbc_file = findDBCFile(id);
  if (!sources_dbc_file) {
    return nullptr;
  }
  auto [_, dbc_file] = *sources_dbc_file;
  return dbc_file->msg(id);
}

const cabana::Msg* DBCManager::msg(uint8_t source, const QString &name) {
  auto sources_dbc_file = findDBCFile({.source = source, .address = 0});
  if (!sources_dbc_file) {
    return nullptr;
  }
  auto [_, dbc_file] = *sources_dbc_file;
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

void DBCManager::updateSources(const SourceSet &s) {
  sources = s;
}

std::optional<std::pair<SourceSet, DBCFile*>> DBCManager::findDBCFile(const MessageId &id) const {
  // Find DBC file that matches id.source, fall back to SOURCE_ALL if no specific DBC is found

  for (auto &[source_set, dbc_file] : dbc_files) {
    if (source_set.contains(id.source)) return {{source_set, dbc_file}};
  }
  for (auto &[source_set, dbc_file] : dbc_files) {
    if (source_set == SOURCE_ALL) return {{sources, dbc_file}};
  }
  return {};

}

DBCManager *dbc() {
  static DBCManager dbc_manager(nullptr);
  return &dbc_manager;
}
