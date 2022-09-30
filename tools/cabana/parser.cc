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
  exit = true;
  thread->quit();
  thread->wait();
}

bool Parser::loadRoute(const QString &route, const QString &data_dir) {
  replay = new Replay(route, {"can", "roadEncodeIdx"}, {}, nullptr, 0, data_dir, this);
  if (!replay->load()) {
    return false;
  }
  replay->start();
  return true;
}

void Parser::openDBC(const QString &name) {
  dbc_name = name;
  dbc = dbc_lookup(name.toStdString());
  for (auto &msg : dbc->msgs) {
    msg_map[msg.address] = &msg;
  }
}

void Parser::process(std::vector<CanData> can) {
  for (auto &data : can) {
    auto &it = items[data.address];
    ++it.cnt;
    it.data = data;
  }
  emit updated();
}

void Parser::recvThread() {
  AlignedBuffer aligned_buf;
  std::unique_ptr<Context> context(Context::create());
  std::unique_ptr<SubSocket> subscriber(SubSocket::create(context.get(), "can"));
  subscriber->setTimeout(100);
  // CANParser parser(1, "toyota_nodsu_pt_generated", true, false);
  while (!exit) {
    std::unique_ptr<Message> msg(subscriber->receive());
    if (!msg) continue;

    capnp::FlatArrayMessageReader cmsg(aligned_buf.align(msg.get()));
    cereal::Event::Reader event = cmsg.getRoot<cereal::Event>();
    // parser.UpdateCans(event.getLogMonoTime(), event.getCan());
    // auto latest = parser.query_latest();
    // for (auto &d : latest) {
    //   qWarning() << d.name.c_str() << d.address << d.value;  // << d.all_values;
    // }
    std::vector<CanData> can;
    can.reserve(event.getCan().size());
    for (const auto &c : event.getCan()) {
      CanData &data = can.emplace_back();
      data.address = c.getAddress();
      data.bus_time = c.getBusTime();
      data.source = c.getSrc();
      data.dat.assign((char *)c.getDat().begin(), c.getDat().size());
    }
    emit received(can);
  }
}

const Msg *Parser::getMsg(uint32_t address) {
  auto it = msg_map.find(address);
  if (it != msg_map.end()) {
    return it->second;
  }
  return nullptr;
}
