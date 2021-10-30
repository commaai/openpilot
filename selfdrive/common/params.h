#pragma once

#include <map>
#include <string>

enum ParamKeyType {
  PERSISTENT = 0x02,
  CLEAR_ON_MANAGER_START = 0x04,
  CLEAR_ON_PANDA_DISCONNECT = 0x08,
  CLEAR_ON_IGNITION_ON = 0x10,
  CLEAR_ON_IGNITION_OFF = 0x20,
  DONT_LOG = 0x40,
  ALL = 0xFFFFFFFF
};

class Params {
public:
  Params(const std::string &path = {});
  bool checkKey(const std::string &key);
  ParamKeyType getKeyType(const std::string &key);
  inline std::string getParamPath(const std::string &key = {}) {
    return key.empty() ? params_path + "/d" : params_path + "/d/" + key;
  }

  // Delete a value
  int remove(const std::string &key);
  void clearAll(ParamKeyType type);

  // helpers for reading values
  std::string get(const std::string &key, bool block = false);
  inline bool getBool(const std::string &key) {
    return get(key) == "1";
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

private:
  std::string params_path;
};
