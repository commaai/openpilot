#pragma once

#include <map>
#include <QList>
#include <QObject>
#include <QString>

#include "tools/cabana/dbc/dbc.h"

const QString AUTO_SAVE_EXTENSION = ".tmp";

class DBCFile : public QObject {
  Q_OBJECT

public:
  DBCFile(const QString &dbc_file_name, QObject *parent=nullptr);
  DBCFile(const QString &name, const QString &content, QObject *parent=nullptr);
  ~DBCFile() {}

  void open(const QString &content);

  bool save();
  bool saveAs(const QString &new_filename);
  bool autoSave();
  bool writeContents(const QString &fn);
  void cleanupAutoSaveFile();
  QString generateDBC();

  cabana::Signal *addSignal(const MessageId &id, const cabana::Signal &sig);
  cabana::Signal *updateSignal(const MessageId &id, const QString &sig_name, const cabana::Signal &sig);
  cabana::Signal *getSignal(const MessageId &id, const QString &sig_name);
  void removeSignal(const MessageId &id, const QString &sig_name);

  void updateMsg(const MessageId &id, const QString &name, uint32_t size, const QString &comment);
  void removeMsg(const MessageId &id);

  QString newMsgName(const MessageId &id);
  QString newSignalName(const MessageId &id);

  const QList<uint8_t>& mask(const MessageId &id) const;

  inline std::map<uint32_t, cabana::Msg> getMessages() const { return msgs; }
  const cabana::Msg *msg(uint32_t address) const;
  const cabana::Msg* msg(const QString &name);
  inline const cabana::Msg *msg(const MessageId &id) const { return msg(id.address); };

  QStringList signalNames() const;
  int signalCount(const MessageId &id) const;
  int signalCount() const;
  inline int msgCount() const { return msgs.size(); }
  inline QString name() const { return name_.isEmpty() ? "untitled" : name_; }
  inline bool isEmpty() const { return (signalCount() == 0) && name_.isEmpty(); }

  QString filename;

private:
  void parseExtraInfo(const QString &content);
  std::map<uint32_t, cabana::Msg> msgs;
  QString name_;
  QList<uint8_t> empty_mask;
};
