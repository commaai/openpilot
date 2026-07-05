#include "tools/loggy/backend/dbc/dbcmanager.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <sstream>
#include <utility>

namespace loggy {

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

  DBCFileChanged();
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

  DBCFileChanged();
  return true;
}

bool DBCManager::assignSources(DBCFile *dbc_file, const SourceSet &sources, std::string *error) {
  if (dbc_file == nullptr) {
    if (error != nullptr) *error = "no DBC file selected";
    return false;
  }
  if (sources.empty()) {
    if (error != nullptr) *error = "no sources parsed";
    return false;
  }

  std::shared_ptr<DBCFile> file;
  for (const auto &[_, candidate] : dbc_files) {
    if (candidate.get() == dbc_file) {
      file = candidate;
      break;
    }
  }
  if (!file) {
    if (error != nullptr) *error = "DBC file is not loaded";
    return false;
  }

  for (auto it = dbc_files.begin(); it != dbc_files.end();) {
    if (it->second.get() == dbc_file) {
      it = dbc_files.erase(it);
    } else {
      ++it;
    }
  }
  for (const int source : sources) {
    dbc_files[source] = file;
  }
  if (error != nullptr) error->clear();
  DBCFileChanged();
  return true;
}

void DBCManager::close(const SourceSet &sources) {
  for (auto s : sources) {
    dbc_files.erase(s);
  }
  DBCFileChanged();
}

void DBCManager::close(DBCFile *dbc_file) {
  for (auto it = dbc_files.begin(); it != dbc_files.end();) {
    if (it->second.get() == dbc_file) {
      it = dbc_files.erase(it);
    } else {
      ++it;
    }
  }
  DBCFileChanged();
}

void DBCManager::closeAll() {
  dbc_files.clear();
  DBCFileChanged();
}

void DBCManager::addSignal(const MessageId &id, const Signal &sig) {
  if (auto m = msg(id)) {
    if (auto s = m->addSignal(sig)) {
      signalAdded(id, s);
      maskUpdated();
    }
  }
}

void DBCManager::updateSignal(const MessageId &id, const std::string &sig_name, const Signal &sig) {
  if (auto m = msg(id)) {
    if (auto s = m->updateSignal(sig_name, sig)) {
      signalUpdated(s);
      maskUpdated();
    }
  }
}

void DBCManager::removeSignal(const MessageId &id, const std::string &sig_name) {
  if (auto m = msg(id)) {
    if (auto s = m->sig(sig_name)) {
      signalRemoved(s);
      m->removeSignal(sig_name);
      maskUpdated();
    }
  }
}

void DBCManager::updateMsg(const MessageId &id, const std::string &name, uint32_t size, const std::string &node, const std::string &comment) {
  auto dbc_file = findDBCFile(id);
  assert(dbc_file);  // This should be impossible
  dbc_file->updateMsg(id, name, size, node, comment);
  msgUpdated(id);
}

void DBCManager::removeMsg(const MessageId &id) {
  auto dbc_file = findDBCFile(id);
  assert(dbc_file);  // This should be impossible
  dbc_file->removeMsg(id);
  msgRemoved(id);
  maskUpdated();
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

const std::map<uint32_t, Msg> &DBCManager::getMessages(uint8_t source) {
  static std::map<uint32_t, Msg> empty_msgs;
  auto dbc_file = findDBCFile(source);
  return dbc_file ? dbc_file->getMessages() : empty_msgs;
}

Msg *DBCManager::msg(const MessageId &id) {
  auto dbc_file = findDBCFile(id);
  return dbc_file ? dbc_file->msg(id) : nullptr;
}

Msg *DBCManager::msg(uint8_t source, const std::string &name) {
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

namespace {

std::string trimSourceSetText(std::string_view text) {
  size_t start = 0;
  while (start < text.size() && std::isspace(static_cast<unsigned char>(text[start]))) ++start;
  size_t end = text.size();
  while (end > start && std::isspace(static_cast<unsigned char>(text[end - 1]))) --end;
  return std::string(text.substr(start, end - start));
}

}  // namespace

bool parseSourceSet(std::string_view text, SourceSet *out, std::string *error) {
  if (out == nullptr) return false;
  SourceSet parsed;
  std::string normalized = trimSourceSetText(text);
  std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  if (normalized.empty() || normalized == "all" || normalized == "*") {
    *out = SOURCE_ALL;
    if (error != nullptr) error->clear();
    return true;
  }

  std::string token;
  std::istringstream stream(normalized);
  while (std::getline(stream, token, ',')) {
    std::istringstream token_stream(token);
    std::string part;
    while (token_stream >> part) {
      char *end = nullptr;
      const long source = std::strtol(part.c_str(), &end, 10);
      if (end == part.c_str() || *end != '\0' || source < 0 || source > 255) {
        if (error != nullptr) *error = "invalid source: " + part;
        return false;
      }
      parsed.insert(static_cast<int>(source));
    }
  }

  if (parsed.empty()) {
    if (error != nullptr) *error = "no sources parsed";
    return false;
  }
  *out = std::move(parsed);
  if (error != nullptr) error->clear();
  return true;
}

}  // namespace loggy
