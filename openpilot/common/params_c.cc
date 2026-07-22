#include "common/params_c.h"

#include <exception>
#include <string>
#include <utility>
#include <vector>

#include "common/params.h"

struct ParamsHandle {
  explicit ParamsHandle(const char *path) : params(path), keys(params.allKeys()) {}

  Params params;
  const std::vector<std::string> keys;
};

namespace {
thread_local std::string last_error;
thread_local std::string result;

ParamsBuffer return_string(std::string value) {
  result = std::move(value);
  return {result.data(), result.size()};
}
}  // namespace

ParamsHandle *params_create(const char *path) {
  try {
    return new ParamsHandle(path);
  } catch (const std::exception &e) {
    last_error = e.what();
    return nullptr;
  }
}

void params_destroy(ParamsHandle *handle) { delete handle; }

const char *params_last_error() { return last_error.c_str(); }

void params_clear_all(ParamsHandle *handle, unsigned int flag) {
  handle->params.clearAll(static_cast<ParamKeyFlag>(flag));
}

bool params_check_key(ParamsHandle *handle, const char *key) {
  return handle->params.checkKey(key);
}

int params_get_key_type(ParamsHandle *handle, const char *key) {
  return handle->params.getKeyType(key);
}

ParamsBuffer params_get_default(ParamsHandle *handle, const char *key) {
  auto value = handle->params.getKeyDefaultValue(key);
  return value.has_value() ? return_string(*value) : ParamsBuffer{nullptr, 0};
}

ParamsBuffer params_get(ParamsHandle *handle, const char *key, bool block) {
  return return_string(handle->params.get(key, block));
}

bool params_get_bool(ParamsHandle *handle, const char *key, bool block) {
  return handle->params.getBool(key, block);
}

int params_put(ParamsHandle *handle, const char *key, const char *value, size_t size, bool block) {
  if (block) return handle->params.put(key, value, size);
  handle->params.putNonBlocking(key, std::string(value, size));
  return 0;
}

int params_put_bool(ParamsHandle *handle, const char *key, bool value, bool block) {
  if (block) return handle->params.putBool(key, value);
  handle->params.putBoolNonBlocking(key, value);
  return 0;
}

int params_remove(ParamsHandle *handle, const char *key) {
  return handle->params.remove(key);
}

ParamsBuffer params_get_path(ParamsHandle *handle, const char *key) {
  return return_string(handle->params.getParamPath(key));
}

size_t params_keys_size(ParamsHandle *handle) { return handle->keys.size(); }

ParamsBuffer params_key_at(ParamsHandle *handle, size_t index) {
  return return_string(handle->keys[index]);
}
