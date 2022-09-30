#pragma once

#include <QApplication>
#include <QObject>
#include <QThread>
#include <atomic>
#include <map>

#include "opendbc/can/common.h"
#include "opendbc/can/common_dbc.h"
#include "tools/replay/replay.h"

struct CanData {
  uint32_t address;
  uint16_t bus_time;
  uint8_t source;
  std::string dat;
};

struct canItem {
  uint64_t cnt;
  CanData data;
};

class Parser : public QObject {
  Q_OBJECT

 public:
  Parser(QObject *parent);
  ~Parser();
  bool loadRoute(const QString &route, const QString &data_dir);
  void openDBC(const QString &name);
  void saveDBC(const QString &name) {}
  const canItem &get(uint32_t address) {
    return items[address];
  }
  const Msg *getMsg(uint32_t address);
 signals:
  void received(std::vector<CanData> can);
  void updated();

public:
  void recvThread();
  void process(std::vector<CanData> can);
  QThread *thread;
  QString dbc_name;
  std::atomic<bool> exit = false;
  std::map<uint32_t, canItem> items;
  Replay *replay = nullptr;
  const DBC *dbc = nullptr;
  std::map<uint32_t, const Msg *> msg_map;
};

Q_DECLARE_METATYPE(std::vector<CanData>);

extern Parser *parser;
