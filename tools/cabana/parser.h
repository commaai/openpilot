#pragma once

#include <QApplication>
#include <QObject>
#include <QThread>
#include <atomic>
#include <map>

#include "opendbc/can/common.h"
#include "opendbc/can/common_dbc.h"
#include "tools/replay/replay.h"

const int DATA_LIST_SIZE = 500;
// const int FPS = 20;

struct CanData {
  QString id;
  double ts;
  uint32_t address;
  uint16_t bus_time;
  uint8_t source;
  QByteArray dat;
  QString hex_dat;
};

class Parser : public QObject {
  Q_OBJECT

 public:
  Parser(QObject *parent);
  ~Parser();
  bool loadRoute(const QString &route, const QString &data_dir, bool use_qcam);
  void openDBC(const QString &name);
  void saveDBC(const QString &name) {}
  void addNewMsg(const Msg &msg);
  void removeSignal(uint32_t address, const QString &sig_name);
  const Msg *getMsg(uint32_t address) {
    auto it = msg_map.find(address);
    return it != msg_map.end() ? it->second : nullptr;
  }
 signals:
  void showPlot(uint32_t address, const QString &name);
  void hidePlot(uint32_t address, const QString &name);
  void received(std::vector<CanData> can);
  void updated();

 public:
  void recvThread();
  void process(std::vector<CanData> can);
  QThread *thread;
  QString dbc_name;
  std::atomic<bool> exit = false;
  std::map<uint32_t, std::list<CanData>> items;
  std::map<uint32_t, uint64_t> counters;
  Replay *replay = nullptr;
  DBC *dbc = nullptr;
  std::map<uint32_t, const Msg *> msg_map;
};

Q_DECLARE_METATYPE(std::vector<CanData>);

extern Parser *parser;
