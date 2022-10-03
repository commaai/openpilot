#include "tools/cabana/parser.h"

#include <QDebug>

#include "cereal/messaging/messaging.h"

Parser::Parser(QObject *parent) : QObject(parent) {
  qRegisterMetaType<std::vector<CanData>>();
  QObject::connect(this, &Parser::received, this, &Parser::process, Qt::QueuedConnection);

  thread = new QThread();
  connect(thread, &QThread::started, [=]() { recvThread(); });
  QObject::connect(thread, &QThread::finished, thread, &QThread::deleteLater);
  thread->start();
}

Parser::~Parser() {
  replay->stop();
  exit = true;
  thread->quit();
  thread->wait();
}

bool Parser::loadRoute(const QString &route, const QString &data_dir, bool use_qcam) {
  replay = new Replay(route, {"can", "roadEncodeIdx"}, {}, nullptr, use_qcam ? REPLAY_FLAG_QCAMERA : 0, data_dir, this);
  if (!replay->load()) {
    return false;
  }
  replay->start();
  return true;
}

void Parser::openDBC(const QString &name) {
  dbc_name = name;
  dbc = const_cast<DBC *>(dbc_lookup(name.toStdString()));
  msg_map.clear();
  for (auto &msg : dbc->msgs) {
    msg_map[msg.address] = &msg;
  }
}

void Parser::process(std::vector<CanData> can) {
  for (auto &data : can) {
    ++counters[data.id];
    auto &list = can_msgs[data.id];
    while (list.size() > DATA_LIST_SIZE) {
      list.pop_front();
    }
    list.push_back(data);
  }
  emit updated();
}

void Parser::recvThread() {
  AlignedBuffer aligned_buf;
  std::unique_ptr<Context> context(Context::create());
  std::unique_ptr<SubSocket> subscriber(SubSocket::create(context.get(), "can"));
  subscriber->setTimeout(100);
  while (!exit) {
    std::unique_ptr<Message> msg(subscriber->receive());
    if (!msg) continue;

    capnp::FlatArrayMessageReader cmsg(aligned_buf.align(msg.get()));
    cereal::Event::Reader event = cmsg.getRoot<cereal::Event>();
    std::vector<CanData> can;
    can.reserve(event.getCan().size());
    for (const auto &c : event.getCan()) {
      CanData &data = can.emplace_back();
      data.address = c.getAddress();
      data.bus_time = c.getBusTime();
      data.source = c.getSrc();
      data.dat.append((char *)c.getDat().begin(), c.getDat().size());
      data.hex_dat = data.dat.toHex(' ').toUpper();
      data.id = QString("%1:%2").arg(data.source).arg(data.address, 1, 16);
      data.ts = (event.getLogMonoTime() - replay->routeStartTime()) / (double)1e6;
    }
    emit received(can);
  }
}

void Parser::addNewMsg(const Msg &msg) {
  dbc->msgs.push_back(msg);
  msg_map[dbc->msgs.back().address] = &dbc->msgs.back();
}

void Parser::removeSignal(const QString &id, const QString &sig_name) {
  Msg *msg = const_cast<Msg *>(getMsg(id));
  if (!msg) return;

  auto it = std::find_if(msg->sigs.begin(), msg->sigs.end(), [=](auto &sig) { return sig_name == sig.name.c_str(); });
  if (it != msg->sigs.end()) {
    msg->sigs.erase(it);
  }
}

uint32_t Parser::addressFromId(const QString &id) {
  return id.mid(id.indexOf(':') + 1).toUInt(nullptr, 16);
}
