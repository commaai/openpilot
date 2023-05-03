#include "tools/cabana/dbc/dbcmanager.h"

#include <numeric>

bool DBCManager::open(SourceSet s, const QString &dbc_file_name, QString *error) {
  try {
    auto all = allDBCFiles();
    auto exist_file = std::find_if(all.begin(), all.end(), [&fn = dbc_file_name](auto &f) { return f->filename == fn; });
    auto file = exist_file != all.end() ? *exist_file : std::make_shared<DBCFile>(dbc_file_name, this);
    for (auto source : s) {
      dbc_files[source].emplace_back(file);
    }
    emit DBCFileChanged();
    return true;
  } catch (std::exception &e) {
    if (error) *error = e.what();
    return false;
  }
}

bool DBCManager::open(SourceSet s, const QString &name, const QString &content, QString *error) {
  try {
    auto file = std::make_shared<DBCFile>(name, content, this);
    for (auto source : s) {
      dbc_files[source].emplace_back(file);
    }
    emit DBCFileChanged();
    return true;
  } catch (std::exception &e) {
    if (error) *error = e.what();
    return false;
  }
}

void DBCManager::close(SourceSet s) {
  for (auto source : s) {
    dbc_files.erase(source);
  }
  emit DBCFileChanged();
}

void DBCManager::closeAll() {
  dbc_files.clear();
  emit DBCFileChanged();
}

void DBCManager::removeSourcesFromFile(DBCFile *file, const SourceSet &source) {
  for (auto &[s, files] : dbc_files) {
    if ((source.empty() || source.contains(s)) && !files.empty()) {
      files.erase(std::find_if(files.begin(), files.end(), [=](auto &f) { return f.get() == file; }));
    }
  }
  emit DBCFileChanged();
}

void DBCManager::addSignal(const MessageId &id, const cabana::Signal &sig) {
  for (auto &f : findDBCFile(id.source)) {
    if (auto m = f->msg(id)) {
      if (cabana::Signal *s = f->addSignal(id, sig)) {
        emit signalAdded(id, s);
        break;
      }
    }
  }
}

void DBCManager::updateSignal(const MessageId &id, const QString &sig_name, const cabana::Signal &sig) {
  for (auto &f : findDBCFile(id.source)) {
    if (auto s = f->updateSignal(id, sig_name, sig)) {
      emit signalUpdated(s);
    }
  }
}

void DBCManager::removeSignal(const MessageId &id, const QString &sig_name) {
  for (auto &f : findDBCFile(id.source)) {
    if (auto s = f->getSignal(id, sig_name)) {
      f->removeSignal(id, sig_name);
      emit signalRemoved(s);
    }
  }
}

void DBCManager::updateMsg(const MessageId &id, const QString &name, uint32_t size) {
  bool updated = false;
  auto &files = findDBCFile(id.source);
  for (auto &f : files) {
    if (auto m = f->msg(id)) {
      f->updateMsg(id, name, size);
      updated = true;
    }
  }

  if (!updated) {
    files[0]->updateMsg(id, name, size);
  }
  emit msgUpdated(id);
}

void DBCManager::removeMsg(const MessageId &id) {
  for (auto &f : findDBCFile(id.source)) {
    f->removeMsg(id);
  }
  emit msgRemoved(id);
}

const std::vector<std::shared_ptr<DBCFile>> &DBCManager::findDBCFile(const uint8_t source) const {
  static std::vector<std::shared_ptr<DBCFile>> empty;
  auto it = dbc_files.find(source);
  if (it == dbc_files.end()) {
    it = dbc_files.find(-1);
  }
  return it == dbc_files.end() ? empty : it->second;
}

const std::set<std::shared_ptr<DBCFile>> DBCManager::allDBCFiles() const {
  std::set<std::shared_ptr<DBCFile>> ret;
  for (auto &[s, files] : dbc_files) {
    ret.insert(files.begin(), files.end());
  }
  return ret;
}

QString DBCManager::newMsgName(const MessageId &id) {
  return QString("NEW_MSG_") + QString::number(id.address, 16).toUpper();
}

QString DBCManager::newSignalName(const MessageId &id) {
  QString name;
  const auto &files = findDBCFile(id.source);
  for (int i = 1, exist = 1; exist; ++i) {
    name = QString("NEW_SIGNAL_%1").arg(i);
    exist = std::any_of(files.begin(), files.end(), [&](auto &f) {
      auto m = f->msg(id);
      return m && m->sig(name) != nullptr;
    });
  }
  return name;
}

std::map<MessageId, cabana::Msg> DBCManager::getMessages(uint8_t source) {
  std::map<MessageId, cabana::Msg> ret;
  for (auto &f : findDBCFile(source)) {
    for (auto &[address, msg] : f->getMessages()) {
      ret[{.source = source, .address = address}] = msg;
    }
  }
  return ret;
}

const cabana::Msg *DBCManager::msg(const MessageId &id) const {
  for (auto &f : findDBCFile(id.source)) {
    if (auto m = f->msg(id)) return m;
  }
  return nullptr;
}

const cabana::Msg *DBCManager::msg(uint8_t source, const QString &name) {
  for (auto &f : findDBCFile(source)) {
    if (auto m = f->msg(name)) return m;
  }
  return nullptr;
}

QStringList DBCManager::signalNames() const {
  QStringList ret;
  for (const auto &f : allDBCFiles()) {
    ret << f->signalNames();
  }
  ret.sort();
  ret.removeDuplicates();
  return ret;
}

int DBCManager::signalCount(const MessageId &id) const {
  const auto &files = findDBCFile(id.source);
  return std::accumulate(files.begin(), files.end(), 0, [&id](int n, auto &f) { return n + f->signalCount(id); });
}

int DBCManager::msgCount() const {
  const auto &files = allDBCFiles();
  return std::accumulate(files.begin(), files.end(), 0, [](int n, auto &f) { return n + f->msgCount(); });
}

int DBCManager::dbcCount(bool no_empty) const {
  const auto &files = allDBCFiles();
  return std::accumulate(files.begin(), files.end(), 0, [=](int n, auto &f) { return n + (!no_empty || !f->isEmpty()); });
}

SourceSet DBCManager::sources(DBCFile *file) const {
  SourceSet ret;
  for (auto &[s, files] : dbc_files) {
    if (std::find_if(files.begin(), files.end(), [file](auto &f) { return f.get() == file; }) != files.end()) {
      ret << s;
    }
  }
  return ret;
}

DBCManager *dbc() {
  static DBCManager dbc_manager(nullptr);
  return &dbc_manager;
}
