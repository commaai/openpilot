#include "tools/cabana/dbc/dbcmanager.h"

#include <algorithm>
#include <cassert>
#include <set>

bool DBCManager::open(const SourceSet &sources, const std::string &dbc_file_name, std::string *error) {
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

  if (callbacks_.file_changed) callbacks_.file_changed();
  return true;
}

bool DBCManager::open(const SourceSet &sources, const std::string &name, const std::string &content, std::string *error) {
  try {
    auto file = std::make_shared<DBCFile>(name, content);
    for (auto s : sources) {
      dbc_files[s] = file;
    }
  } catch (std::exception &e) {
    if (error) *error = e.what();
    return false;
  }

  if (callbacks_.file_changed) callbacks_.file_changed();
  return true;
}

void DBCManager::close(const SourceSet &sources) {
  for (auto s : sources) {
    dbc_files[s] = nullptr;
  }
  if (callbacks_.file_changed) callbacks_.file_changed();
}

void DBCManager::close(DBCFile *dbc_file) {
  for (auto &[_, f] : dbc_files) {
    if (f.get() == dbc_file) f = nullptr;
  }
  if (callbacks_.file_changed) callbacks_.file_changed();
}

void DBCManager::closeAll() {
  dbc_files.clear();
  if (callbacks_.file_changed) callbacks_.file_changed();
}

void DBCManager::addSignal(const MessageId &id, const cabana::Signal &sig) {
  if (auto m = msg(id)) {
    if (auto s = m->addSignal(sig)) {
      if (callbacks_.signal_added) callbacks_.signal_added(id, s);
      if (callbacks_.mask_updated) callbacks_.mask_updated();
    }
  }
}

void DBCManager::updateSignal(const MessageId &id, const std::string &sig_name, const cabana::Signal &sig) {
  if (auto m = msg(id)) {
    if (auto s = m->updateSignal(sig_name, sig)) {
      if (callbacks_.signal_updated) callbacks_.signal_updated(s);
      if (callbacks_.mask_updated) callbacks_.mask_updated();
    }
  }
}

void DBCManager::removeSignal(const MessageId &id, const std::string &sig_name) {
  if (auto m = msg(id)) {
    if (auto s = m->sig(sig_name)) {
      if (callbacks_.signal_removed) callbacks_.signal_removed(s);
      m->removeSignal(sig_name);
      if (callbacks_.mask_updated) callbacks_.mask_updated();
    }
  }
}

void DBCManager::updateMsg(const MessageId &id, const std::string &name, uint32_t size, const std::string &node, const std::string &comment) {
  auto dbc_file = findDBCFile(id);
  assert(dbc_file);  // This should be impossible
  dbc_file->updateMsg(id, name, size, node, comment);
  if (callbacks_.msg_updated) callbacks_.msg_updated(id);
}

void DBCManager::removeMsg(const MessageId &id) {
  auto dbc_file = findDBCFile(id);
  assert(dbc_file);  // This should be impossible
  dbc_file->removeMsg(id);
  if (callbacks_.msg_removed) callbacks_.msg_removed(id);
  if (callbacks_.mask_updated) callbacks_.mask_updated();
}

std::string DBCManager::newMsgName(const MessageId &id) {
  char buf[64];
  snprintf(buf, sizeof(buf), "NEW_MSG_%X", id.address);
  return buf;
}

std::string DBCManager::newSignalName(const MessageId &id) {
  auto m = msg(id);
  return m ? m->newSignalName() : "";
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

cabana::Msg *DBCManager::msg(uint8_t source, const std::string &name) {
  auto dbc_file = findDBCFile(source);
  return dbc_file ? dbc_file->msg(name) : nullptr;
}

std::vector<std::string> DBCManager::signalNames() {
  // Used for autocompletion
  std::set<std::string> names;
  for (auto &f : allDBCFiles()) {
    for (auto &[_, m] : f->getMessages()) {
      for (auto sig : m.getSignals()) {
        names.insert(sig->name);
      }
    }
  }
  std::vector<std::string> ret(names.begin(), names.end());
  std::sort(ret.begin(), ret.end());
  return ret;
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

std::string toString(const SourceSet &ss) {
  std::string result;
  for (int source : ss) {
    if (!result.empty()) result += ", ";
    result += (source == -1) ? "all" : std::to_string(source);
  }
  return result;
}

DBCManager *dbc() {
  static DBCManager dbc_manager;
  return &dbc_manager;
}
