#include "tools/cabana/dbc/dbcmanager.h"
#include <algorithm>

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
  auto it = std::find_if(dbc_files.begin(), dbc_files.end(), [&](auto &f) { return f.first == s; });
  if (it != dbc_files.end()) {
    delete it->second;
    dbc_files.erase(it);
    emit DBCFileChanged();
  }
}

void DBCManager::close(DBCFile *dbc_file) {
  auto it = std::find_if(dbc_files.begin(), dbc_files.end(), [=](auto &f) { return f.second == dbc_file; });
  if (it != dbc_files.end()) {
    delete it->second;
    dbc_files.erase(it);
    emit DBCFileChanged();
  }
}

void DBCManager::closeAll() {
  for (auto &[_, f] : dbc_files) delete f;
  dbc_files.clear();
  emit DBCFileChanged();
}

void DBCManager::removeSourcesFromFile(DBCFile *dbc_file, SourceSet s) {
  auto it = std::find_if(dbc_files.begin(), dbc_files.end(), [=](auto &f) { return f.second == dbc_file; });
  if (it != dbc_files.end()) {
    if (it->first == SOURCE_ALL) it->first = sources;
    it->first -= s;
    if (it->first.empty()) {
      delete it->second;
      dbc_files.erase(it);
    }
    emit DBCFileChanged();
  }
}

void DBCManager::addSignal(const MessageId &id, const cabana::Signal &sig) {
  if (auto m = msg(id)) {
    if (auto s = m->addSignal(sig)) {
      emit signalAdded(id, s);
    }
  }
}

void DBCManager::updateSignal(const MessageId &id, const QString &sig_name, const cabana::Signal &sig) {
  if (auto m = msg(id)) {
    if (auto s = m->updateSignal(sig_name, sig)) {
      emit signalUpdated(s);
    }
  }
}

void DBCManager::removeSignal(const MessageId &id, const QString &sig_name) {
  if (auto m = msg(id)) {
    if (auto s = m->sig(sig_name)) {
      emit signalRemoved(s);
      m->removeSignal(sig_name);
    }
  }
}

void DBCManager::updateMsg(const MessageId &id, const QString &name, uint32_t size, const QString &comment) {
  auto dbc_file = findDBCFile(id);
  assert(dbc_file); // This should be impossible
  dbc_file->updateMsg(id, name, size, comment);
  emit msgUpdated(id);
}

void DBCManager::removeMsg(const MessageId &id) {
  auto dbc_file = findDBCFile(id);
  assert(dbc_file); // This should be impossible
  dbc_file->removeMsg(id);
  emit msgRemoved(id);
}

QString DBCManager::newMsgName(const MessageId &id) {
  return QString("NEW_MSG_") + QString::number(id.address, 16).toUpper();
}

QString DBCManager::newSignalName(const MessageId &id) {
  auto m = msg(id);
  return m ? m->newSignalName() : "";
}

const QList<uint8_t> &DBCManager::mask(const MessageId &id) {
  static QList<uint8_t> empty_mask;
  auto m = msg(id);
  return m ? m->mask : empty_mask;
}

const std::map<uint32_t, cabana::Msg> &DBCManager::getMessages(uint8_t source) {
  static std::map<uint32_t, cabana::Msg> empty_msgs;
  auto dbc_file = findDBCFile(source);
  return dbc_file ? dbc_file->getMessages() : empty_msgs;
}

cabana::Msg *DBCManager::msg(const MessageId &id) {
  auto dbc_file = findDBCFile(id);
  return dbc_file ? dbc_file->msg(id) : nullptr;
}

cabana::Msg* DBCManager::msg(uint8_t source, const QString &name) {
  auto dbc_file = findDBCFile(source);
  return dbc_file ? dbc_file->msg(name) : nullptr;
}

QStringList DBCManager::signalNames() {
  // Used for autocompletion
  QStringList ret;
  for (auto &[_, dbc_file] : dbc_files) {
    for (auto &[_, m] : dbc_file->getMessages()) {
       for (auto sig : m.getSignals()) {
          ret << sig->name;
       }
    }
  }
  ret.sort();
  ret.removeDuplicates();
  return ret;
}

int DBCManager::signalCount(const MessageId &id) {
  auto m = msg(id);
  return m ? m->sigs.size() : 0;
}

int DBCManager::signalCount() {
  return std::accumulate(dbc_files.cbegin(), dbc_files.cend(), 0,
                         [](int &n, auto &f) { return n + f.second->signalCount(); });
}

int DBCManager::msgCount() {
  return std::accumulate(dbc_files.cbegin(), dbc_files.cend(), 0,
                         [](int &n, auto &f) { return n + f.second->msgCount(); });
}

int DBCManager::dbcCount() {
  return dbc_files.size();
}

int DBCManager::nonEmptyDBCCount() {
  return std::count_if(dbc_files.cbegin(), dbc_files.cend(), [](auto &f) { return !f.second->isEmpty(); });
}

void DBCManager::updateSources(const SourceSet &s) {
  sources = s;
}

DBCFile *DBCManager::findDBCFile(const uint8_t source) {
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
