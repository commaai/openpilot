#pragma once

#include <map>
#include <string>

#include "tools/loggy/backend/dbc/dbc.h"

namespace loggy {

class DBCFile {
public:
  DBCFile(const std::string &dbc_file_name);
  DBCFile(const std::string &name, const std::string &content);
  ~DBCFile() {}

  bool save();
  bool save_as(const std::string &new_filename);
  bool write_contents(const std::string &fn);
  std::string generate_dbc();

  void update_msg(const MessageId &id, const std::string &name, uint32_t size, const std::string &node, const std::string &comment);
  inline void remove_msg(const MessageId &id) { msgs.erase(id.address); }

  inline const std::map<uint32_t, Msg> &messages() const { return msgs; }
  Msg *msg(uint32_t address);
  Msg *msg(const std::string &name);
  inline Msg *msg(const MessageId &id) { return msg(id.address); }
  Signal *signal(uint32_t address, const std::string &name);

  inline std::string name() const { return name_.empty() ? "untitled" : name_; }
  inline bool is_empty() const { return msgs.empty() && name_.empty(); }

  std::string filename;

private:
  void parse(const std::string &content);
  Msg *parse_bo(const std::string &line);
  void parse_sg(const std::string &line, Msg *current_msg, int &multiplexor_cnt);
  void parse_cm_bo(const std::string &line, const std::string &content, size_t line_offset);
  void parse_cm_sg(const std::string &line, const std::string &content, size_t line_offset);
  void parse_val(const std::string &line);

  std::string header;
  std::map<uint32_t, Msg> msgs;
  std::string name_;
};

}  // namespace loggy
