#include "tools/cabana/dbc/dbcmanager.h"

#include <algorithm>
#include <numeric>

bool DBCManager::open(const SourceSet &sources, const QString &dbc_file_name, QString *error) {
  try {
    auto it = std::find_if(dbc_files.begin(), dbc_files.end(), [&](auto &f) { return f.second->filename == dbc_file_name; });
    auto file = (it != dbc_files.end()) ? it->second : std::make_shared<DBCFile>(dbc_file_name, this);
    for (auto s : sources) {
      dbc_files[s] = file;
    }
  } catch (std::exception &e) {
    if (error) *error = e.what();
    return false;
  }

  emit DBCFileChanged();
  return true;
}

bool DBCManager::open(const SourceSet &sources, const QString &name, const QString &content, QString *error) {
  try {
    auto file = std::make_shared<DBCFile>(name, content, this);
    for (auto s : sources) {
      dbc_files[s] = file;
    }
  } catch (std::exception &e) {
    if (error) *error = e.what();
    return false;
  }

  emit DBCFileChanged();
  return true;
}

void DBCManager::close(const SourceSet &sources) {
  for (auto s : sources) dbc_files.erase(s);
  emit DBCFileChanged();
}

void DBCManager::close(DBCFile *dbc_file) {
  for (auto it = dbc_files.begin(); it != dbc_files.end(); /**/) {
    it = (it->second.get() == dbc_file) ? dbc_files.erase(it) : ++it;
  }
  emit DBCFileChanged();
}

void DBCManager::closeAll() {
  dbc_files.clear();
  emit DBCFileChanged();
}

void DBCManager::addSignal(const MessageId &id, const cabana::Signal &sig) {
  if (cabana::Signal *s = findDBCFile(id)->addSignal(id, sig)) {
    emit signalAdded(id, s);
  }
}

void DBCManager::updateSignal(const MessageId &id, const QString &sig_name, const cabana::Signal &sig) {
  if (cabana::Signal *s = findDBCFile(id)->updateSignal(id, sig_name, sig)) {
    emit signalUpdated(s);
  }
}

void DBCManager::removeSignal(const MessageId &id, const QString &sig_name) {
  auto dbc_file = findDBCFile(id);
  if (auto s = dbc_file->getSignal(id, sig_name)) {
    emit signalRemoved(s);
    dbc_file->removeSignal(id, sig_name);
  }
}

void DBCManager::updateMsg(const MessageId &id, const QString &name, uint32_t size, const QString &comment) {
  findDBCFile(id)->updateMsg(id, name, size, comment);
  emit msgUpdated(id);
}

void DBCManager::removeMsg(const MessageId &id) {
  findDBCFile(id)->removeMsg(id);
  emit msgRemoved(id);
}

QStringList DBCManager::signalNames() const {
  QStringList ret;
  for (auto &[_, dbc_file] : dbc_files) ret << dbc_file->signalNames();
  ret.sort();
  ret.removeDuplicates();
  return ret;
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
  auto it = dbc_files.count(source) ? dbc_files.find(source) : dbc_files.find(-1);
  if (it == dbc_files.end()) {
    // Ensure always have at least one file open
    it = dbc_files.insert({-1, std::make_shared<DBCFile>("", "")}).first;
  }
  return it->second.get();
}

const SourceSet DBCManager::sources(const DBCFile *dbc_file) const {
  SourceSet sources;
  for (auto &[s, _] : dbc_files) sources.insert(s);
  return sources;
}

std::set<DBCFile *> DBCManager::allDBCFiles() const {
  std::set<DBCFile *> files;
  for (auto &[_, f] : dbc_files) files.insert(f.get());
  return files;
}

DBCManager *dbc() {
  static DBCManager dbc_manager(nullptr);
  return &dbc_manager;
}
