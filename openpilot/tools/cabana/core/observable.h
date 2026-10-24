#pragma once

#include <functional>
#include <map>
#include <memory>
#include <utility>
#include <vector>

namespace observable_detail {
struct HandlerTable {
  virtual ~HandlerTable() = default;
  virtual void erase(int id) = 0;
};
}  // namespace observable_detail

// disconnects on destruction; safe to outlive the Observable
class Connection {
public:
  Connection() = default;
  Connection(std::weak_ptr<observable_detail::HandlerTable> table, int id) : table_(std::move(table)), id_(id) {}
  Connection(Connection &&other) noexcept { *this = std::move(other); }
  Connection &operator=(Connection &&other) noexcept {
    if (this != &other) {
      disconnect();
      table_ = std::move(other.table_);
      id_ = std::exchange(other.id_, -1);
    }
    return *this;
  }
  Connection(const Connection &) = delete;
  Connection &operator=(const Connection &) = delete;
  ~Connection() { disconnect(); }

  void disconnect() {
    if (auto table = table_.lock()) table->erase(id_);
    table_.reset();
    id_ = -1;
  }

private:
  std::weak_ptr<observable_detail::HandlerTable> table_;
  int id_ = -1;
};

using Connections = std::vector<Connection>;

// main thread only. handlers may disconnect (or destroy the Observable) while being invoked.
template <typename... Args>
class Observable {
public:
  using Handler = std::function<void(Args...)>;

  Observable() = default;
  Observable(const Observable &) = delete;
  Observable &operator=(const Observable &) = delete;

  [[nodiscard]] Connection connect(Handler handler) {
    int id = table_->next_id++;
    table_->handlers.emplace(id, std::make_shared<Handler>(std::move(handler)));
    return Connection(table_, id);
  }

  void operator()(Args... args) const {
    auto table = table_;
    std::vector<int> ids;
    ids.reserve(table->handlers.size());
    for (const auto &[id, _] : table->handlers) ids.push_back(id);
    for (int id : ids) {
      auto it = table->handlers.find(id);
      if (it == table->handlers.end()) continue;
      auto handler = it->second;
      (*handler)(args...);
    }
  }

private:
  struct Table : observable_detail::HandlerTable {
    std::map<int, std::shared_ptr<Handler>> handlers;
    int next_id = 0;
    void erase(int id) override { handlers.erase(id); }
  };
  std::shared_ptr<Table> table_ = std::make_shared<Table>();
};
