#include <time.h>
#include <assert.h>
#include <stdlib.h>
#include <string>
#include <mutex>

#include "cereal/services.h"
#include "cereal/messaging/messaging.h"


const bool SIMULATION = (getenv("SIMULATION") != nullptr) && (std::string(getenv("SIMULATION")) == "1");

static inline uint64_t nanos_since_boot() {
  struct timespec t;
  clock_gettime(CLOCK_BOOTTIME, &t);
  return t.tv_sec * 1000000000ULL + t.tv_nsec;
}

static inline bool inList(const std::vector<const char *> &list, const char *value) {
  for (auto &v : list) {
    if (strcmp(value, v) == 0) return true;
  }
  return false;
}

class MessageContext {
public:
  MessageContext() : ctx_(nullptr) {}
  ~MessageContext() { delete ctx_; }
  inline Context *context() {
    std::call_once(init_flag, [=]() { ctx_ = Context::create(); });
    return ctx_;
  }
private:
  Context *ctx_;
  std::once_flag init_flag;
};

MessageContext message_context;

struct SubMaster::SubMessage {
  std::string name;
  SubSocket *socket = nullptr;
  int freq = 0;
  bool updated = false, alive = false, valid = true, ignore_alive;
  uint64_t rcv_time = 0, rcv_frame = 0;
  void *allocated_msg_reader = nullptr;
  bool is_polled = false;
  capnp::FlatArrayMessageReader *msg_reader = nullptr;
  AlignedBuffer aligned_buf;
  cereal::Event::Reader event;
};

SubMaster::SubMaster(const std::vector<const char *> &service_list, const std::vector<const char *> &poll,
                     const char *address, const std::vector<const char *> &ignore_alive) {
  poller_ = Poller::create();
  for (auto name : service_list) {
    assert(services.count(std::string(name)) > 0);

    service serv = services.at(std::string(name));
    SubSocket *socket = SubSocket::create(message_context.context(), name, address ? address : "127.0.0.1", true);
    assert(socket != 0);
    bool is_polled = inList(poll, name) || poll.empty();
    if (is_polled) poller_->registerSocket(socket);
    SubMessage *m = new SubMessage{
      .name = name,
      .socket = socket,
      .freq = serv.frequency,
      .ignore_alive = inList(ignore_alive, name),
      .allocated_msg_reader = malloc(sizeof(capnp::FlatArrayMessageReader)),
      .is_polled = is_polled};
    m->msg_reader = new (m->allocated_msg_reader) capnp::FlatArrayMessageReader({});
    messages_[socket] = m;
    services_[name] = m;
  }
}

void SubMaster::update(int timeout) {
  for (auto &kv : messages_) kv.second->updated = false;

  auto sockets = poller_->poll(timeout);

  // add non-polled sockets for non-blocking receive
  for (auto &kv : messages_) {
    SubMessage *m = kv.second;
    SubSocket *s = kv.first;
    if (!m->is_polled) sockets.push_back(s);
  }

  uint64_t current_time = nanos_since_boot();

  std::vector<std::pair<std::string, cereal::Event::Reader>> messages;

  for (auto s : sockets) {
    Message *msg = s->receive(true);
    if (msg == nullptr) continue;

    SubMessage *m = messages_.at(s);

    m->msg_reader->~FlatArrayMessageReader();
    capnp::ReaderOptions options;
    options.traversalLimitInWords = kj::maxValue; // Don't limit
    m->msg_reader = new (m->allocated_msg_reader) capnp::FlatArrayMessageReader(m->aligned_buf.align(msg), options);
    delete msg;
    messages.push_back({m->name, m->msg_reader->getRoot<cereal::Event>()});
  }

  update_msgs(current_time, messages);
}

void SubMaster::update_msgs(uint64_t current_time, const std::vector<std::pair<std::string, cereal::Event::Reader>> &messages){
  if (++frame == UINT64_MAX) frame = 1;

  for (auto &kv : messages) {
    auto m_find = services_.find(kv.first);
    if (m_find == services_.end()){
      continue;
    }
    SubMessage *m = m_find->second;
    m->event = kv.second;
    m->updated = true;
    m->rcv_time = current_time;
    m->rcv_frame = frame;
    m->valid = m->event.getValid();
    if (SIMULATION) m->alive = true;
  }

  if (!SIMULATION) {
    for (auto &kv : messages_) {
      SubMessage *m = kv.second;
      m->alive = (m->freq <= (1e-5) || ((current_time - m->rcv_time) * (1e-9)) < (10.0 / m->freq));
    }
  }
}

bool SubMaster::all_(const std::vector<const char *> &service_list, bool valid, bool alive) {
  int found = 0;
  for (auto &kv : messages_) {
    SubMessage *m = kv.second;
    if (service_list.size() == 0 || inList(service_list, m->name.c_str())) {
      found += (!valid || m->valid) && (!alive || (m->alive || m->ignore_alive));
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

bool SubMaster::alive(const char *name) const {
  return services_.at(name)->alive;
}

bool SubMaster::valid(const char *name) const {
  return services_.at(name)->valid;
}

uint64_t SubMaster::rcv_frame(const char *name) const {
  return services_.at(name)->rcv_frame;
}

uint64_t SubMaster::rcv_time(const char *name) const {
  return services_.at(name)->rcv_time;
}

cereal::Event::Reader &SubMaster::operator[](const char *name) const {
  return services_.at(name)->event;
}

SubMaster::~SubMaster() {
  delete poller_;
  for (auto &kv : messages_) {
    SubMessage *m = kv.second;
    m->msg_reader->~FlatArrayMessageReader();
    free(m->allocated_msg_reader);
    delete m->socket;
    delete m;
  }
}

PubMaster::PubMaster(const std::vector<const char *> &service_list) {
  for (auto name : service_list) {
    assert(services.count(name) > 0);
    PubSocket *socket = PubSocket::create(message_context.context(), name);
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
