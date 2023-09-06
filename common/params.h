#pragma once

#include <future>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "common/queue.h"

enum ParamKeyType {
  PERSISTENT = 0x02,
  CLEAR_ON_MANAGER_START = 0x04,
  CLEAR_ON_ONROAD_TRANSITION = 0x08,
  CLEAR_ON_OFFROAD_TRANSITION = 0x10,
  DONT_LOG = 0x20,
  ALL = 0xFFFFFFFF
};

class Params {
public:
  Params(const std::string &path = {});
  std::vector<std::string> allKeys() const;
  bool checkKey(const std::string &key);
  ParamKeyType getKeyType(const std::string &key);
  inline std::string getParamPath(const std::string &key = {}) {
    return params_path + prefix + (key.empty() ? "" : "/" + key);
  }

  // Delete a value
  int remove(const std::string &key);
  void clearAll(ParamKeyType type);

  // helpers for reading values
  std::string get(const std::string &key, bool block = false);
  inline bool getBool(const std::string &key, bool block = false) {
    return get(key, block) == "1";
  }
  std::map<std::string, std::string> readAll();

  // helpers for writing values
  int put(const char *key, const char *val, size_t value_size);
  inline int put(const std::string &key, const std::string &val) {
    return put(key.c_str(), val.data(), val.size());
  }
  inline int putBool(const std::string &key, bool val) {
    return put(key.c_str(), val ? "1" : "0", 1);
  }
  void putNonBlocking(const std::string &key, const std::string &val);
  inline void putBoolNonBlocking(const std::string &key, bool val) {
    putNonBlocking(key, val ? "1" : "0");
  }

private:
  std::string params_path;
  std::string prefix;
};

class AsyncWriter {
public:
  AsyncWriter() {}
  ~AsyncWriter();
  void queue(const std::tuple<std::string, std::string, std::string> &dat);

private:
  void write();

  std::future<void> future;
  SafeQueue<std::tuple<std::string, std::string, std::string>> q;
};
