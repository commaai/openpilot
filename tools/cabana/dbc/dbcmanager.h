#pragma once

#include <QObject>
#include <memory>
#include <map>
#include <set>
#include <vector>

#include "tools/cabana/dbc/dbcfile.h"

typedef std::set<int> SourceSet;
const SourceSet SOURCE_ALL = {-1};
const int INVALID_SOURCE = 0xff;
inline bool operator<(const std::shared_ptr<DBCFile> &l, const std::shared_ptr<DBCFile> &r) { return l.get() < r.get(); }

class DBCManager : public QObject {
  Q_OBJECT

public:
  DBCManager(QObject *parent) : QObject(parent) {}
  ~DBCManager() {}
  bool open(const SourceSet &sources, const QString &dbc_file_name, QString *error = nullptr);
  bool open(const SourceSet &sources, const QString &name, const QString &content, QString *error = nullptr);
  void close(const SourceSet &sources);
  void close(DBCFile *dbc_file);
  void closeAll();

  void addSignal(const MessageId &id, const cabana::Signal &sig);
  void updateSignal(const MessageId &id, const QString &sig_name, const cabana::Signal &sig);
  void removeSignal(const MessageId &id, const QString &sig_name);

  void updateMsg(const MessageId &id, const QString &name, uint32_t size, const QString &node, const QString &comment);
  void removeMsg(const MessageId &id);

  QString newMsgName(const MessageId &id);
  QString newSignalName(const MessageId &id);
  const std::vector<uint8_t>& mask(const MessageId &id);

  const std::map<uint32_t, cabana::Msg> &getMessages(uint8_t source);
  cabana::Msg *msg(const MessageId &id);
  cabana::Msg* msg(uint8_t source, const QString &name);

  QStringList signalNames();
  int signalCount(const MessageId &id);
  int signalCount();
  int msgCount();
  int dbcCount();
  int nonEmptyDBCCount();

  const SourceSet sources(const DBCFile *dbc_file) const;
  DBCFile *findDBCFile(const uint8_t source);
  inline DBCFile *findDBCFile(const MessageId &id) { return findDBCFile(id.source); }
  std::set<DBCFile *> allDBCFiles();

signals:
  void signalAdded(MessageId id, const cabana::Signal *sig);
  void signalRemoved(const cabana::Signal *sig);
  void signalUpdated(const cabana::Signal *sig);
  void msgUpdated(MessageId id);
  void msgRemoved(MessageId id);
  void DBCFileChanged();
  void maskUpdated();

private:
  std::map<int, std::shared_ptr<DBCFile>> dbc_files;
};

DBCManager *dbc();

QString toString(const SourceSet &ss);
inline QString msgName(const MessageId &id) {
  auto msg = dbc()->msg(id);
  return msg ? msg->name : UNTITLED;
}
