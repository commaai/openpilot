#pragma once

#include <map>
#include <QList>
#include <QMetaType>
#include <QObject>
#include <QString>
#include <QSet>
#include <QDebug>

#include "tools/cabana/dbc.h"

class DBCFile : public QObject {
  Q_OBJECT

public:
  DBCFile(const QString &dbc_file_name, QObject *parent=nullptr);
  DBCFile(const QString &name, const QString &content, QObject *parent=nullptr);
  ~DBCFile() {}

  void open(const QString &name, const QString &content);
  QString generateDBC();

  cabana::Signal *addSignal(const MessageId &id, const cabana::Signal &sig);
  cabana::Signal *updateSignal(const MessageId &id, const QString &sig_name, const cabana::Signal &sig);
  cabana::Signal *getSignal(const MessageId &id, const QString &sig_name);
  void removeSignal(const MessageId &id, const QString &sig_name);

  void updateMsg(const MessageId &id, const QString &name, uint32_t size);
  void removeMsg(const MessageId &id);

  std::map<uint32_t, cabana::Msg> getMessages();
  const cabana::Msg *msg(const MessageId &id) const;
  const cabana::Msg *msg(uint32_t address) const;
  const cabana::Msg* msg(const QString &name);
  QStringList signalNames() const;
  int msgCount() const;
  QString name() const;

private:
  void parseExtraInfo(const QString &content);
  std::map<uint32_t, cabana::Msg> msgs;
  QString name_;
};
