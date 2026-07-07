#pragma once

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "tools/cabana/dbc/dbcfile.h"
#include "tools/cabana/utils/event.h"

typedef std::set<int> SourceSet;
const SourceSet SOURCE_ALL = {-1};
const int INVALID_SOURCE = 0xff;
inline bool operator<(const std::shared_ptr<DBCFile> &l, const std::shared_ptr<DBCFile> &r) { return l.get() < r.get(); }

class DBCManager {
public:
  DBCManager() = default;
  ~DBCManager() = default;
  bool open(const SourceSet &sources, const std::string &dbc_file_name, std::string *error = nullptr);
  bool open(const SourceSet &sources, const std::string &name, const std::string &content, std::string *error = nullptr);
  void close(const SourceSet &sources);
  void close(DBCFile *dbc_file);
  void closeAll();

  void addSignal(const MessageId &id, const cabana::Signal &sig);
  void updateSignal(const MessageId &id, const std::string &sig_name, const cabana::Signal &sig);
  void removeSignal(const MessageId &id, const std::string &sig_name);

  void updateMsg(const MessageId &id, const std::string &name, uint32_t size, const std::string &node, const std::string &comment);
  void removeMsg(const MessageId &id);

  std::string newMsgName(const MessageId &id);
  std::string newSignalName(const MessageId &id);

  const std::map<uint32_t, cabana::Msg> &getMessages(uint8_t source);
  cabana::Msg *msg(const MessageId &id);
  cabana::Msg* msg(uint8_t source, const std::string &name);

  std::vector<std::string> signalNames();
  inline int dbcCount() { return allDBCFiles().size(); }
  int nonEmptyDBCCount();

  const SourceSet sources(const DBCFile *dbc_file) const;
  DBCFile *findDBCFile(const uint8_t source);
  inline DBCFile *findDBCFile(const MessageId &id) { return findDBCFile(id.source); }
  std::set<DBCFile *> allDBCFiles();

  cabana::Event<MessageId, const cabana::Signal *> signalAdded;
  cabana::Event<const cabana::Signal *> signalRemoved;
  cabana::Event<const cabana::Signal *> signalUpdated;
  cabana::Event<MessageId> msgUpdated;
  cabana::Event<MessageId> msgRemoved;
  cabana::Event<> DBCFileChanged;
  cabana::Event<> maskUpdated;

private:
  std::map<int, std::shared_ptr<DBCFile>> dbc_files;
};

DBCManager *dbc();

std::string toString(const SourceSet &ss);
inline std::string msgName(const MessageId &id) {
  auto msg = dbc()->msg(id);
  return msg ? msg->name : UNTITLED;
}
