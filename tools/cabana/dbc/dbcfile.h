#pragma once

#include <map>
#include <string>
#include <QTextStream>

#include "tools/cabana/dbc/dbc.h"

class DBCFile {
public:
  DBCFile(const std::string &dbc_file_name);
  DBCFile(const std::string &name, const std::string &content);
  ~DBCFile() {}

  bool save();
  bool saveAs(const std::string &new_filename);
  bool writeContents(const std::string &fn);
  std::string generateDBC();

  void updateMsg(const MessageId &id, const std::string &name, uint32_t size, const std::string &node, const std::string &comment);
  inline void removeMsg(const MessageId &id) { msgs.erase(id.address); }

  inline const std::map<uint32_t, cabana::Msg> &getMessages() const { return msgs; }
  cabana::Msg *msg(uint32_t address);
  cabana::Msg *msg(const std::string &name);
  inline cabana::Msg *msg(const MessageId &id) { return msg(id.address); }
  cabana::Signal *signal(uint32_t address, const std::string &name);

  inline std::string name() const { return name_.empty() ? "untitled" : name_; }
  inline bool isEmpty() const { return msgs.empty() && name_.empty(); }

  std::string filename;

private:
  void parse(const QString &content);
  cabana::Msg *parseBO(const QString &line);
  void parseSG(const QString &line, cabana::Msg *current_msg, int &multiplexor_cnt);
  void parseCM_BO(const QString &line, const QString &content, const QString &raw_line, const QTextStream &stream);
  void parseCM_SG(const QString &line, const QString &content, const QString &raw_line, const QTextStream &stream);
  void parseVAL(const QString &line);

  std::string header;
  std::map<uint32_t, cabana::Msg> msgs;
  std::string name_;
};
