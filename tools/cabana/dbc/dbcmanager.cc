#include "tools/cabana/dbc/dbcmanager.h"

#include <algorithm>
#include <numeric>

bool DBCManager::open(const SourceSet &sources, const QString &dbc_file_name, QString *error) {
  try {
    auto it = std::find_if(dbc_files.begin(), dbc_files.end(),
                           [&](auto &f) { return f.second && f.second->filename == dbc_file_name; });
    auto file = (it != dbc_files.end()) ? it->second : std::make_shared<DBCFile>(dbc_file_name);
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
    auto file = std::make_shared<DBCFile>(name, content);
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
  for (auto s : sources) {
    dbc_files[s] = nullptr;
  }
  emit DBCFileChanged();
}

void DBCManager::close(DBCFile *dbc_file) {
  for (auto &[_, f] : dbc_files) {
    if (f.get() == dbc_file) f = nullptr;
  }
  emit DBCFileChanged();
}

void DBCManager::closeAll() {
  dbc_files.clear();
  emit DBCFileChanged();
}

void DBCManager::addSignal(const MessageId &id, const cabana::Signal &sig) {
  if (auto m = msg(id)) {
    if (auto s = m->addSignal(sig)) {
      emit signalAdded(id, s);
      emit maskUpdated();
    }
  }
}

void DBCManager::updateSignal(const MessageId &id, const QString &sig_name, const cabana::Signal &sig) {
  if (auto m = msg(id)) {
    if (auto s = m->updateSignal(sig_name, sig)) {
      emit signalUpdated(s);
      emit maskUpdated();
    }
  }
}

void DBCManager::removeSignal(const MessageId &id, const QString &sig_name) {
  if (auto m = msg(id)) {
    if (auto s = m->sig(sig_name)) {
      emit signalRemoved(s);
      m->removeSignal(sig_name);
      emit maskUpdated();
    }
  }
}

void DBCManager::updateMsg(const MessageId &id, const QString &name, uint32_t size, const QString &node, const QString &comment) {
  auto dbc_file = findDBCFile(id);
  assert(dbc_file);  // This should be impossible
  dbc_file->updateMsg(id, name, size, node, comment);
  emit msgUpdated(id);
}

void DBCManager::removeMsg(const MessageId &id) {
  auto dbc_file = findDBCFile(id);
  assert(dbc_file);  // This should be impossible
  dbc_file->removeMsg(id);
  emit msgRemoved(id);
  emit maskUpdated();
}

QString DBCManager::newMsgName(const MessageId &id) {
  return QString("NEW_MSG_") + QString::number(id.address, 16).toUpper();
}

QString DBCManager::newSignalName(const MessageId &id) {
  auto m = msg(id);
  return m ? m->newSignalName() : "";
}

const std::vector<uint8_t> &DBCManager::mask(const MessageId &id) {
  static std::vector<uint8_t> empty_mask;
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

cabana::Msg *DBCManager::msg(uint8_t source, const QString &name) {
  auto dbc_file = findDBCFile(source);
  return dbc_file ? dbc_file->msg(name) : nullptr;
}

QStringList DBCManager::signalNames() {
  // Used for autocompletion
  QStringList ret;
  for (auto &f : allDBCFiles()) {
    for (auto &[_, m] : f->getMessages()) {
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
  auto files = allDBCFiles();
  return std::accumulate(files.cbegin(), files.cend(), 0, [](int &n, auto &f) { return n + f->signalCount(); });
}

int DBCManager::msgCount() {
  auto files = allDBCFiles();
  return std::accumulate(files.cbegin(), files.cend(), 0, [](int &n, auto &f) { return n + f->msgCount(); });
}

int DBCManager::dbcCount() {
  return allDBCFiles().size();
}

int DBCManager::nonEmptyDBCCount() {
  auto files = allDBCFiles();
  return std::count_if(files.cbegin(), files.cend(), [](auto &f) { return !f->isEmpty(); });
}

DBCFile *DBCManager::findDBCFile(const uint8_t source) {
  // Find DBC file that matches id.source, fall back to SOURCE_ALL if no specific DBC is found
  auto it = dbc_files.count(source) ? dbc_files.find(source) : dbc_files.find(-1);
  return it != dbc_files.end() ? it->second.get() : nullptr;
}

std::set<DBCFile *> DBCManager::allDBCFiles() {
  std::set<DBCFile *> files;
  for (const auto &[_, f] : dbc_files) {
    if (f) files.insert(f.get());
  }
  return files;
}

const SourceSet DBCManager::sources(const DBCFile *dbc_file) const {
  SourceSet sources;
  for (auto &[s, f] : dbc_files) {
    if (f.get() == dbc_file) sources.insert(s);
  }
  return sources;
}

QString toString(const SourceSet &ss) {
  return std::accumulate(ss.cbegin(), ss.cend(), QString(), [](QString str, int source) {
    if (!str.isEmpty()) str += ", ";
    return str + (source == -1 ? QStringLiteral("all") : QString::number(source));
  });
}

DBCManager *dbc() {
  static DBCManager dbc_manager(nullptr);
  return &dbc_manager;
}
