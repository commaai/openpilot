#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

enum ParamKeyFlag {
  PERSISTENT = 0x02,
  CLEAR_ON_MANAGER_START = 0x04,
  CLEAR_ON_ONROAD_TRANSITION = 0x08,
  CLEAR_ON_OFFROAD_TRANSITION = 0x10,
  DONT_LOG = 0x20,
  DEVELOPMENT_ONLY = 0x40,
  CLEAR_ON_IGNITION_ON = 0x80,
  WAIT_FOR_DISK = 0x100,  // a write or remove only returns once it's on the disk
  ALL = 0xFFFFFFFF
};

enum ParamKeyType {
  STRING = 0, // must be utf-8 decodable
  BOOL = 1,
  INT = 2,
  FLOAT = 3,
  TIME = 4, // ISO 8601
  JSON = 5,
  BYTES = 6
};

struct ParamKeyAttributes {
  uint32_t flags;
  ParamKeyType type;
  std::optional<std::string> default_value = std::nullopt;
};

class Params {
public:
  explicit Params(const std::string &path = {});
  ~Params();
  // Not copyable.
  Params(const Params&) = delete;
  Params& operator=(const Params&) = delete;

  std::vector<std::string> allKeys() const;
  bool checkKey(const std::string &key);
  ParamKeyFlag getKeyFlag(const std::string &key);
  ParamKeyType getKeyType(const std::string &key);
  std::optional<std::string> getKeyDefaultValue(const std::string &key);
  inline std::string getParamPath(const std::string &key = {}) {
    return params_path + params_prefix + (key.empty() ? "" : "/" + key);
  }

  // Delete a value
  int remove(const std::string &key);
  void clearAll(ParamKeyFlag flag);

  // helpers for reading values
  std::string get(const std::string &key, bool block = false);
  inline bool getBool(const std::string &key, bool block = false) {
    return get(key, block) == "1";
  }
  std::map<std::string, std::string> readAll();

  // helpers for writing values
  // on the device the params are in RAM, so nothing here waits for the disk unless block is set
  int put(const char *key, const char *val, size_t value_size, bool block = false);
  inline int put(const std::string &key, const std::string &val, bool block = false) {
    return put(key.c_str(), val.data(), val.size(), block);
  }
  inline int putBool(const std::string &key, bool val, bool block = false) {
    return put(key.c_str(), val ? "1" : "0", 1, block);
  }

  // copy what changed in RAM to the disk
  int save();
  // for whoever calls save(): returns when a param was written, or after the timeout
  void waitForWrite(int timeout_ms);

private:
  std::string params_path;
  std::string params_prefix;
  std::string disk_path;  // empty if params_path is the disk
  std::map<std::string, std::pair<uint64_t, int64_t>> saved;  // the files save() knows are on the disk
  int watch_fd = -1;
};
