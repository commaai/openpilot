#pragma once

#include <QApplication>
#include <QObject>
#include <QThread>
#include <atomic>
#include <map>

#include "opendbc/can/common.h"
#include "opendbc/can/common_dbc.h"
#include "tools/replay/replay.h"

const int DATA_LIST_SIZE = 50;
const int FPS = 20;

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
  static uint32_t addressFromId(const QString &id);
  bool loadRoute(const QString &route, const QString &data_dir, bool use_qcam);
  void openDBC(const QString &name);
  void saveDBC(const QString &name) {}
  void addNewMsg(const Msg &msg);
  void removeSignal(const QString &id, const QString &sig_name);
  const Msg *getMsg(const QString &id) {
    return getMsg(addressFromId(id));
  }
  const Msg *getMsg(uint32_t address) {
    auto it = msg_map.find(address);
    return it != msg_map.end() ? it->second : nullptr;
  }
  const Signal *getSig(const QString &id, const QString &sig_name);
  void setRange(double min, double max);
  inline std::pair<double, double> range() const { return {begin_sec, end_sec}; }
  inline double currentSec() const {return current_sec; }
  void resetRange();
  // inline bool isZoomed() { return begin_sec != 0; } const;

 signals:
  void showPlot(const QString &id, const QString &name);
  void hidePlot(const QString &id, const QString &name);
  void signalRemoved(const QString &id, const QString &sig_name);
  void eventsMerged();
  void rangeChanged(double min, double max);
  void received(std::vector<CanData> can);
  void updated();

public:
  std::map<QString, std::list<CanData>> can_msgs;
  std::map<QString, uint64_t> counters;
  Replay *replay = nullptr;


protected:
  void recvThread();
  void process(std::vector<CanData> can);
  void segmentsMerged();

  double current_sec = 0.;
  std::atomic<bool> exit = false;
  QThread *thread;
  QString dbc_name;
  double begin_sec = 0;
  double end_sec = 0;
  double event_begin_sec = 0;
  double event_end_sec = 0;
  bool is_zoomed = false;
  DBC *dbc = nullptr;
  std::map<uint32_t, const Msg *> msg_map;
};

Q_DECLARE_METATYPE(std::vector<CanData>);

extern Parser *parser;
