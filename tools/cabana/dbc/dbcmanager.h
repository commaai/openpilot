#pragma once

#include <memory>
#include <set>

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

  void updateMsg(const MessageId &id, const QString &name, uint32_t size, const QString &comment);
  void removeMsg(const MessageId &id);

  inline QString newMsgName(const MessageId &id) const { return findDBCFile(id)->newMsgName(id); }
  inline QString newSignalName(const MessageId &id) const { return findDBCFile(id)->newSignalName(id); }

  inline const QList<uint8_t> &mask(const MessageId &id) const { return findDBCFile(id)->mask(id); }
  const SourceSet sources(const DBCFile *dbc_file) const;

  inline const std::map<uint32_t, cabana::Msg> &getMessages(uint8_t source) const { return findDBCFile(source)->getMessages(); }
  inline const cabana::Msg *msg(const MessageId &id) const { return findDBCFile(id)->msg(id); }
  inline const cabana::Msg *msg(uint8_t source, const QString &name) const { return findDBCFile(source)->msg(name); }

  QStringList signalNames() const;
  inline int signalCount(const MessageId &id) const { return findDBCFile(id)->signalCount(id); }
  int signalCount() const;
  int msgCount() const;
  inline int dbcCount() const { return dbc_files.size(); }
  int nonEmptyDBCCount() const;

  DBCFile *findDBCFile(const uint8_t source) const;
  inline DBCFile *findDBCFile(const MessageId &id) const { return findDBCFile(id.source); }
  std::set<DBCFile *> allDBCFiles() const;

signals:
  void signalAdded(MessageId id, const cabana::Signal *sig);
  void signalRemoved(const cabana::Signal *sig);
  void signalUpdated(const cabana::Signal *sig);
  void msgUpdated(MessageId id);
  void msgRemoved(MessageId id);
  void DBCFileChanged();

private:
  mutable std::map<int, std::shared_ptr<DBCFile>> dbc_files;
};

DBCManager *dbc();

inline QString msgName(const MessageId &id) {
  auto msg = dbc()->msg(id);
  return msg ? msg->name : UNTITLED;
}

inline QString toString(const SourceSet &ss) {
  QStringList ret;
  for (auto s : ss) {
    ret << (s == -1 ? QString("all") : QString::number(s));
  }
  return ret.join(", ");
}
