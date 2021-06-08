#pragma once

#include <map>
#include <sstream>
#include <string>

#define ERR_NO_VALUE -33

enum ParamKeyType {
  PERSISTENT = 0x02,
  CLEAR_ON_MANAGER_START = 0x04,
  CLEAR_ON_PANDA_DISCONNECT = 0x08,
  CLEAR_ON_IGNITION_ON = 0x10,
  CLEAR_ON_IGNITION_OFF = 0x20,
  ALL = 0x02 | 0x04 | 0x08 | 0x10 | 0x20
};

class Params {
private:
  std::string params_path;

public:
  Params(bool persistent_param = false);
  Params(const std::string &path);

  bool checkKey(const std::string &key);

  // Delete a value
  int remove(const char *key);
  inline int remove(const std::string &key) {
    return remove (key.c_str());
  }
  void clearAll(ParamKeyType type);

  // read all values
  int readAll(std::map<std::string, std::string> *params);

  // helpers for reading values
  std::string get(const char *key, bool block = false);

  inline std::string get(const std::string &key, bool block = false) {
    return get(key.c_str(), block);
  }

  inline std::string getParamsPath() {
    return params_path;
  }

  template <class T>
  std::optional<T> get(const char *key, bool block = false) {
    std::istringstream iss(get(key, block));
    T value{};
    iss >> value;
    return iss.fail() ? std::nullopt : std::optional(value);
  }

  inline bool getBool(const std::string &key) {
    return getBool(key.c_str());
  }

  inline bool getBool(const char *key) {
    return get(key) == "1";
  }

  // helpers for writing values
  int put(const char* key, const char* val, size_t value_size);

  inline int put(const std::string &key, const std::string &val) {
    return put(key.c_str(), val.data(), val.size());
  }

  inline int putBool(const char *key, bool val) {
    return put(key, val ? "1" : "0", 1);
  }

  inline int putBool(const std::string &key, bool val) {
    return putBool(key.c_str(), val);
  }
};
