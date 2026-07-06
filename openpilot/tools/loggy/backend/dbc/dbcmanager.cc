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

bool DBCManager::open(const SourceSet &sources, const std::string &dbc_file_name, std::string &error) {
  try {
    auto it = std::find_if(dbc_files.begin(), dbc_files.end(),
                           [&](auto &f) { return f.second && f.second->filename == dbc_file_name; });
    auto file = (it != dbc_files.end()) ? it->second : std::make_shared<DBCFile>(dbc_file_name);
    for (auto s : sources) {
      dbc_files[s] = file;
    }
  } catch (std::exception &e) {
    error = e.what();
    return false;
  }

  return true;
}

bool DBCManager::open(const SourceSet &sources, const std::string &name, const std::string &content, std::string &error) {
  try {
    auto file = std::make_shared<DBCFile>(name, content);
    for (auto s : sources) {
      dbc_files[s] = file;
    }
  } catch (std::exception &e) {
    error = e.what();
    return false;
  }

  return true;
}

bool DBCManager::assign_sources(DBCFile *dbc_file, const SourceSet &sources, std::string &error) {
  if (dbc_file == nullptr) {
    error = "no DBC file selected";
    return false;
  }
  if (sources.empty()) {
    error = "no sources parsed";
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
    error = "DBC file is not loaded";
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
  error.clear();
  return true;
}

void DBCManager::close(const SourceSet &sources) {
  for (auto s : sources) {
    dbc_files.erase(s);
  }
}

void DBCManager::close(DBCFile *dbc_file) {
  for (auto it = dbc_files.begin(); it != dbc_files.end();) {
    if (it->second.get() == dbc_file) {
      it = dbc_files.erase(it);
    } else {
      ++it;
    }
  }
}

void DBCManager::close_all() {
  dbc_files.clear();
}

void DBCManager::add_signal(const MessageId &id, const Signal &sig) {
  if (auto m = msg(id)) {
    m->add_signal(sig);
  }
}

void DBCManager::updateSignal(const MessageId &id, const std::string &sig_name, const Signal &sig) {
  if (auto m = msg(id)) {
    m->updateSignal(sig_name, sig);
  }
}

void DBCManager::removeSignal(const MessageId &id, const std::string &sig_name) {
  if (auto m = msg(id)) {
    m->removeSignal(sig_name);
  }
}

void DBCManager::update_msg(const MessageId &id, const std::string &name, uint32_t size, const std::string &node, const std::string &comment) {
  auto dbc_file = find_dbc_file(id);
  assert(dbc_file);  // This should be impossible
  dbc_file->update_msg(id, name, size, node, comment);
}

void DBCManager::remove_msg(const MessageId &id) {
  auto dbc_file = find_dbc_file(id);
  assert(dbc_file);  // This should be impossible
  dbc_file->remove_msg(id);
}

std::string DBCManager::new_msg_name(const MessageId &id) {
  char buf[64];
  snprintf(buf, sizeof(buf), "NEW_MSG_%X", id.address);
  return buf;
}

std::string DBCManager::new_signal_name(const MessageId &id) {
  auto m = msg(id);
  return m ? m->new_signal_name() : "";
}

const std::map<uint32_t, Msg> &DBCManager::messages(uint8_t source) {
  auto dbc_file = find_dbc_file(source);
  return dbc_file ? dbc_file->messages() : empty_msgs_;
}

Msg *DBCManager::msg(const MessageId &id) {
  auto dbc_file = find_dbc_file(id);
  return dbc_file ? dbc_file->msg(id) : nullptr;
}

Msg *DBCManager::msg(uint8_t source, const std::string &name) {
  auto dbc_file = find_dbc_file(source);
  return dbc_file ? dbc_file->msg(name) : nullptr;
}

std::vector<std::string> DBCManager::signal_names() {
  // Used for autocompletion
  std::set<std::string> names;
  for (auto &f : all_dbc_files()) {
    for (auto &[_, m] : f->messages()) {
      for (auto sig : m.signals()) {
        names.insert(sig->name);
      }
    }
  }
  std::vector<std::string> ret(names.begin(), names.end());
  std::sort(ret.begin(), ret.end());
  return ret;
}

int DBCManager::non_empty_dbc_count() {
  auto files = all_dbc_files();
  return std::count_if(files.cbegin(), files.cend(), [](auto &f) { return !f->is_empty(); });
}

DBCFile *DBCManager::find_dbc_file(const uint8_t source) {
  // Find DBC file that matches id.source, fall back to SOURCE_ALL if no specific DBC is found
  auto it = dbc_files.count(source) ? dbc_files.find(source) : dbc_files.find(-1);
  return it != dbc_files.end() ? it->second.get() : nullptr;
}

std::set<DBCFile *> DBCManager::all_dbc_files() {
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

std::string to_string(const SourceSet &ss) {
  std::string result;
  for (int source : ss) {
    if (!result.empty()) result += ", ";
    result += (source == -1) ? "all" : std::to_string(source);
  }
  return result;
}

namespace {

std::string trimSourceSetText(std::string_view text) {
  size_t start_ = 0;
  while (start_ < text.size() && std::isspace(static_cast<unsigned char>(text[start_]))) ++start_;
  size_t end = text.size();
  while (end > start_ && std::isspace(static_cast<unsigned char>(text[end - 1]))) --end;
  return std::string(text.substr(start_, end - start_));
}

}  // namespace

bool parse_source_set(std::string_view text, SourceSet &out, std::string &error) {
  SourceSet parsed;
  std::string normalized = trimSourceSetText(text);
  std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  if (normalized.empty() || normalized == "all" || normalized == "*") {
    out = SOURCE_ALL;
    error.clear();
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
        error = "invalid source: " + part;
        return false;
      }
      parsed.insert(static_cast<int>(source));
    }
  }

  if (parsed.empty()) {
    error = "no sources parsed";
    return false;
  }
  out = std::move(parsed);
  error.clear();
  return true;
}

}  // namespace loggy
