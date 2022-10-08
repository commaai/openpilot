#pragma once

#include <QObject>

#include "opendbc/can/common.h"
#include "opendbc/can/common_dbc.h"

class DBCManager : public QObject {
  Q_OBJECT

public:
  DBCManager(QObject *parent);
  ~DBCManager();

  void open(const QString &dbc_file_name);
  void save(const QString &dbc_file_name);

  const Signal *getSig(const QString &id, const QString &sig_name) const;
  void addSignal(const QString &id, const Signal &sig);
  void updateSignal(const QString &id, const QString &sig_name, const Signal &sig);
  void removeSignal(const QString &id, const QString &sig_name);

  void updateMsg(const QString &id, const QString &name, uint32_t size);

  static uint32_t addressFromId(const QString &id);
  inline static std::vector<std::string> allDBCNames() { return get_dbc_names(); }
  inline QString name() const { return dbc_name; }
  inline const Msg *msg(const QString &id) const { return msg(addressFromId(id)); }
  inline const Msg *msg(uint32_t address) const {
    auto it = msg_map.find(address);
    return it != msg_map.end() ? it->second : nullptr;
  }

signals:
  void signalAdded(const QString &id, const QString &sig_name);
  void signalRemoved(const QString &id, const QString &sig_name);
  void signalUpdated(const QString &id, const QString &sig_name);
  void DBCFileChanged();


protected:
  QString dbc_name;
  DBC *dbc = nullptr;
  std::map<uint32_t, const Msg *> msg_map;
};

// TODO: Add helper function in dbc.h
double get_raw_value(uint8_t *data, size_t data_size, const Signal &sig);
int bigEndianStartBitsIndex(int start_bit);
int bigEndianBitIndex(int index);

// A global pointer referring to the unique DBCManager object
DBCManager *dbc();
