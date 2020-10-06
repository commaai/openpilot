#include <assert.h>
#include <time.h>
#include "messaging.hpp"
#include "services.h"

static inline uint64_t nanos_since_boot() {
  struct timespec t;
  clock_gettime(CLOCK_BOOTTIME, &t);
  return t.tv_sec * 1000000000ULL + t.tv_nsec;
}

static const service *get_service(const char *name) {
  for (const auto &it : services) {
    if (strcmp(it.name, name) == 0) return &it;
  }
  return nullptr;
}

static inline bool inList(const std::initializer_list<const char *> &list, const char *value) {
  for (auto &v : list) {
    if (strcmp(value, v) == 0) return true;
  }
  return false;
}

class MessageContext {
public:
  MessageContext() { ctx_ = Context::create(); }
  ~MessageContext() { delete ctx_; }
  Context *ctx_;
};
MessageContext ctx;

struct SubMaster::SubMessage {
  std::string name;
  SubSocket *socket = nullptr;
  int freq = 0;
  bool updated = false, alive = false, valid = false, ignore_alive;
  uint64_t rcv_time = 0, rcv_frame = 0;
  void *allocated_msg_reader = nullptr;
  capnp::FlatArrayMessageReader *msg_reader = nullptr;
  kj::Array<capnp::word> buf;
  cereal::Event::Reader event;
};

SubMaster::SubMaster(const std::initializer_list<const char *> &service_list, const char *address,
                     const std::initializer_list<const char *> &ignore_alive) {
  poller_ = Poller::create();
  for (auto name : service_list) {
    const service *serv = get_service(name);
    assert(serv != nullptr);
    SubSocket *socket = SubSocket::create(ctx.ctx_, name, address ? address : "127.0.0.1", true);
    assert(socket != 0);
    poller_->registerSocket(socket);
    SubMessage *m = new SubMessage{
      .socket = socket,
      .freq = serv->frequency,
      .ignore_alive = inList(ignore_alive, name),
      .allocated_msg_reader = malloc(sizeof(capnp::FlatArrayMessageReader)),
      .buf = kj::heapArray<capnp::word>(1024)};
    messages_[socket] = m;
    services_[name] = m;
  }
}

int SubMaster::update(int timeout) {
  if (++frame == UINT64_MAX) frame = 1;
  for (auto &kv : messages_) kv.second->updated = false;

  int updated = 0;
  auto sockets = poller_->poll(timeout);
  uint64_t current_time = nanos_since_boot();
  for (auto s : sockets) {
    Message *msg = s->receive(true);
    if (msg == nullptr) continue;

    SubMessage *m = messages_.at(s);
    const size_t size = (msg->getSize() / sizeof(capnp::word)) + 1;
    if (m->buf.size() < size) {
      m->buf = kj::heapArray<capnp::word>(size);
    }
    memcpy(m->buf.begin(), msg->getData(), msg->getSize());
    delete msg;

    if (m->msg_reader) {
      m->msg_reader->~FlatArrayMessageReader();
    }
    m->msg_reader = new (m->allocated_msg_reader) capnp::FlatArrayMessageReader(kj::ArrayPtr<capnp::word>(m->buf.begin(), size));
    m->event = m->msg_reader->getRoot<cereal::Event>();
    m->updated = true;
    m->rcv_time = current_time;
    m->rcv_frame = frame;
    m->valid = m->event.getValid();

    ++updated;
  }

  for (auto &kv : messages_) {
    SubMessage *m = kv.second;
    m->alive = (m->freq <= (1e-5) || ((current_time - m->rcv_time) * (1e-9)) < (10.0 / m->freq));
  }
  return updated;
}

bool SubMaster::all_(const std::initializer_list<const char *> &service_list, bool valid, bool alive) {
  int found = 0;
  for (auto &kv : messages_) {
    SubMessage *m = kv.second;
    if (service_list.size() == 0 || inList(service_list, m->name.c_str())) {
      found += (!valid || m->valid) && (!alive || (m->alive && !m->ignore_alive));
    }
  }
  return service_list.size() == 0 ? found == messages_.size() : found == service_list.size();
}

void SubMaster::drain() {
  while (true) {
    auto polls = poller_->poll(0);
    if (polls.size() == 0)
      break;

    for (auto sock : polls) {
      Message *msg = sock->receive(true);
      delete msg;
    }
  }
}

bool SubMaster::updated(const char *name) const {
  return services_.at(name)->updated;
}

uint64_t SubMaster::rcv_frame(const char *name) const {
  return services_.at(name)->rcv_frame;
}

cereal::Event::Reader &SubMaster::operator[](const char *name) {
  return services_.at(name)->event;
};

SubMaster::~SubMaster() {
  delete poller_;
  for (auto &kv : messages_) {
    SubMessage *m = kv.second;
    if (m->msg_reader) {
      m->msg_reader->~FlatArrayMessageReader();
    }
    free(m->allocated_msg_reader);
    delete m->socket;
    delete m;
  }
}

PubMaster::PubMaster(const std::initializer_list<const char *> &service_list) {
  for (auto name : service_list) {
    assert(get_service(name) != nullptr);
    PubSocket *socket = PubSocket::create(ctx.ctx_, name);
    assert(socket);
    sockets_[name] = socket;
  }
}

int PubMaster::send(const char *name, MessageBuilder &msg) {
  auto bytes = msg.toBytes();
  return send(name, bytes.begin(), bytes.size());
}

PubMaster::~PubMaster() {
  for (auto s : sockets_) delete s.second;
}
