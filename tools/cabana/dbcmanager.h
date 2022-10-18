#pragma once

#include <QObject>

#include "opendbc/can/common_dbc.h"

class DBCManager : public QObject {
  Q_OBJECT

public:
  DBCManager(QObject *parent);
  ~DBCManager();

  void open(const QString &dbc_file_name);
  void open(const QString &name, const QString &content);
  void save(const QString &dbc_file_name);

  void addSignal(const QString &id, const Signal &sig);
  void updateSignal(const QString &id, const QString &sig_name, const Signal &sig);
  void removeSignal(const QString &id, const QString &sig_name);

  static uint32_t addressFromId(const QString &id);
  inline static std::vector<std::string> allDBCNames() { return get_dbc_names(); }
  inline QString name() const { return dbc_name; }

  void updateMsg(const QString &id, const QString &name, uint32_t size);
  inline const Msg *msg(const QString &id) const { return msg(addressFromId(id)); }
  inline const Msg *msg(uint32_t address) const {
    auto it = msg_map.find(address);
    return it != msg_map.end() ? it->second : nullptr;
  }

signals:
  void signalAdded(const Signal *sig);
  void signalRemoved(const Signal *sig);
  void signalUpdated(const Signal *sig);
  void msgUpdated(const QString &id);
  void DBCFileChanged();

private:
  QString dbc_name;
  DBC *dbc = nullptr;
  std::unordered_map<uint32_t, const Msg *> msg_map;
};

// TODO: Add helper function in dbc.h
double get_raw_value(uint8_t *data, size_t data_size, const Signal &sig);
int bigEndianStartBitsIndex(int start_bit);
int bigEndianBitIndex(int index);

DBCManager *dbc();
