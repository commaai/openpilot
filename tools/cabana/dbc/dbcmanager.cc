#include "tools/cabana/dbc/dbcmanager.h"

#include <algorithm>
#include <numeric>

bool DBCManager::open(SourceSet s, const QString &dbc_file_name, QString *error) {
  for (int i = 0; i < dbc_files.size(); i++) {
    auto [ss, dbc_file] = dbc_files[i];

    // Check if file is already open, and merge sources
    if (dbc_file->filename == dbc_file_name) {
      dbc_files[i] = {ss | s, dbc_file};

      emit DBCFileChanged();
      return true;
    }
  }

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
  try {
    dbc_files.push_back({s, new DBCFile(name, content, this)});
  } catch (std::exception &e) {
    if (error) *error = e.what();
    return false;
  }

  emit DBCFileChanged();
  return true;
}

void DBCManager::close(SourceSet s) {
  // removing the ones that match the sourceset
  auto it = std::find_if(dbc_files.begin(), dbc_files.end(), [&s](auto &f) { return f.first == s; });
  if (it != dbc_files.end()) {
    delete it->second;
    dbc_files.erase(it);
    emit DBCFileChanged();
  }
}

void DBCManager::close(DBCFile *dbc_file) {
  // removing the one that matches dbc_file*
  assert(dbc_file != nullptr);
  auto it = std::find_if(dbc_files.begin(), dbc_files.end(), [dbc_file](auto &f) { return f.second == dbc_file; });
  if (it != dbc_files.end()) {
    delete it->second;
    dbc_files.erase(it);
    emit DBCFileChanged();
  }
}

void DBCManager::closeAll() {
  for (auto &f : dbc_files) delete f.second;
  dbc_files.clear();
  emit DBCFileChanged();
}

void DBCManager::removeSourcesFromFile(DBCFile *dbc_file, SourceSet s) {
  // for the given dbc_file* remove s from the current sources
  assert(dbc_file != nullptr);
  auto it = std::find_if(dbc_files.begin(), dbc_files.end(), [&](auto &f) { return f.second == dbc_file && f.first.contains(s); });
  if (it != dbc_files.end()) {
    it->first -= s;
    if (it->first.empty()) {
      delete it->second;
      dbc_files.erase(it);
    }
    emit DBCFileChanged();
  }
}

void DBCManager::addSignal(const MessageId &id, const cabana::Signal &sig) {
  auto dbc_file = findDBCFile(id);
  assert(dbc_file);  // Create new DBC?
  if (cabana::Signal *s = dbc_file->addSignal(id, sig)) {
    emit signalAdded(id, s);
  }
}

void DBCManager::updateSignal(const MessageId &id, const QString &sig_name, const cabana::Signal &sig) {
  auto dbc_file = findDBCFile(id);
  assert(dbc_file);  // This should be impossible
  if (cabana::Signal *s = dbc_file->updateSignal(id, sig_name, sig)) {
    emit signalUpdated(s);
  }
}

void DBCManager::removeSignal(const MessageId &id, const QString &sig_name) {
  auto dbc_file = findDBCFile(id);
  assert(dbc_file);  // This should be impossible
  if (cabana::Signal *s = dbc_file->getSignal(id, sig_name)) {
    emit signalRemoved(s);
    dbc_file->removeSignal(id, sig_name);
  }
}

void DBCManager::updateMsg(const MessageId &id, const QString &name, uint32_t size, const QString &comment) {
  auto dbc_file = findDBCFile(id);
  assert(dbc_file);  // This should be impossible
  dbc_file->updateMsg(id, name, size, comment);
  emit msgUpdated(id);
}

void DBCManager::removeMsg(const MessageId &id) {
  auto dbc_file = findDBCFile(id);
  assert(dbc_file);  // This should be impossible
  dbc_file->removeMsg(id);
  emit msgRemoved(id);
}

QString DBCManager::newMsgName(const MessageId &id) {
  auto dbc_file = findDBCFile(id);
  assert(dbc_file);  // This should be impossible
  return dbc_file->newMsgName(id);
}

QString DBCManager::newSignalName(const MessageId &id) {
  auto dbc_file = findDBCFile(id);
  assert(dbc_file);  // This should be impossible
  return dbc_file->newSignalName(id);
}

const QList<uint8_t> &DBCManager::mask(const MessageId &id) const {
  auto dbc_file = findDBCFile(id);
  return dbc_file ? dbc_file->mask(id) : empty_mask;
}

std::map<MessageId, cabana::Msg> DBCManager::getMessages(uint8_t source) {
  std::map<MessageId, cabana::Msg> ret;
  if (auto dbc_file = findDBCFile(source)) {
    for (auto &[address, msg] : dbc_file->getMessages()) {
      MessageId id = {.source = source, .address = address};
      ret[id] = msg;
    }
  }
  return ret;
}

const cabana::Msg *DBCManager::msg(const MessageId &id) const {
  auto dbc_file = findDBCFile(id);
  return dbc_file ? dbc_file->msg(id) : nullptr;
}

const cabana::Msg *DBCManager::msg(uint8_t source, const QString &name) const {
  auto dbc_file = findDBCFile(source);
  return dbc_file ? dbc_file->msg(name) : nullptr;
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

int DBCManager::signalCount(const MessageId &id) const {
  auto dbc_file = findDBCFile(id);
  return dbc_file ? dbc_file->signalCount(id) : 0;
}

int DBCManager::signalCount() const {
  return std::accumulate(dbc_files.cbegin(), dbc_files.cend(), 0, [](int &n, auto &f) { return n + f.second->signalCount(); });
}

int DBCManager::msgCount() const {
  return std::accumulate(dbc_files.cbegin(), dbc_files.cend(), 0, [](int &n, auto &f) { return n + f.second->msgCount(); });
}

int DBCManager::nonEmptyDBCCount() const {
  return std::count_if(dbc_files.cbegin(), dbc_files.cend(), [](auto &f) { return !f.second->isEmpty(); });
}

DBCFile *DBCManager::findDBCFile(const uint8_t source) const {
  // Find DBC file that matches id.source, fall back to SOURCE_ALL if no specific DBC is found
  for (auto &[source_set, dbc_file] : dbc_files) {
    if (source_set.contains(source)) return dbc_file;
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
