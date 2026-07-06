#pragma once

#include <map>
#include <memory>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "tools/loggy/backend/dbc/dbcfile.h"
namespace loggy {

typedef std::set<int> SourceSet;
const SourceSet SOURCE_ALL = {-1};
const int INVALID_SOURCE = 0xff;

class DBCManager {
public:
  DBCManager() = default;
  ~DBCManager() = default;
  bool open(const SourceSet &sources, const std::string &dbc_file_name, std::string &error);
  bool open(const SourceSet &sources, const std::string &name, const std::string &content, std::string &error);
  bool assign_sources(DBCFile *dbc_file, const SourceSet &sources, std::string &error);
  void close(const SourceSet &sources);
  void close(DBCFile *dbc_file);
  void close_all();

  void add_signal(const MessageId &id, const Signal &sig);
  void update_signal(const MessageId &id, const std::string &sig_name, const Signal &sig);
  void remove_signal(const MessageId &id, const std::string &sig_name);

  void update_msg(const MessageId &id, const std::string &name, uint32_t size, const std::string &node, const std::string &comment);
  void remove_msg(const MessageId &id);

  std::string new_msg_name(const MessageId &id);
  std::string new_signal_name(const MessageId &id);

  const std::map<uint32_t, Msg> &messages(uint8_t source);
  Msg *msg(const MessageId &id);
  Msg* msg(uint8_t source, const std::string &name);

  std::vector<std::string> signal_names();
  inline int dbc_count() { return all_dbc_files().size(); }
  int non_empty_dbc_count();

  const SourceSet sources(const DBCFile *dbc_file) const;
  DBCFile *find_dbc_file(const uint8_t source);
  inline DBCFile *find_dbc_file(const MessageId &id) { return find_dbc_file(id.source); }
  std::set<DBCFile *> all_dbc_files();

private:
  std::map<uint32_t, Msg> empty_msgs_;
  std::map<int, std::shared_ptr<DBCFile>> dbc_files;
};

bool parse_source_set(std::string_view text, SourceSet &out, std::string &error);
std::string to_string(const SourceSet &ss);
inline std::string msg_name(DBCManager &manager, const MessageId &id) {
  auto msg = manager.msg(id);
  return msg ? msg->name : UNTITLED;
}

}  // namespace loggy
