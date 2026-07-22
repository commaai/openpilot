#include <cstddef>
#include <cstdio>
#include <exception>
#include <string>
#include <utility>
#include <vector>

#include "common/params.h"

typedef struct {
  const char *data;
  size_t size;
} ParamsBuffer;

struct ParamsHandle {
  ParamsHandle(const char *path, size_t path_size) : params(std::string(path, path_size)), keys(params.allKeys()) {}

  Params params;
  const std::vector<std::string> keys;
};

namespace {
thread_local char last_error[512] = {};
thread_local std::string result;

ParamsBuffer return_string(std::string value) {
  result = std::move(value);
  return {result.data(), result.size()};
}

template <typename Result, typename Callable>
Result guard(Callable &&callable, Result failure) noexcept {
  last_error[0] = '\0';
  try {
    return callable();
  } catch (const std::exception &e) {
    snprintf(last_error, sizeof(last_error), "%s", e.what());
  } catch (...) {
    snprintf(last_error, sizeof(last_error), "unknown C++ exception");
  }
  return failure;
}

template <typename Callable>
void guard(Callable &&callable) noexcept {
  guard([&]() {
    callable();
    return true;
  }, false);
}
}  // namespace

extern "C" {

ParamsHandle *params_create(const char *path, size_t path_size) {
  return guard([&]() { return new ParamsHandle(path, path_size); }, static_cast<ParamsHandle *>(nullptr));
}

void params_destroy(ParamsHandle *handle) { guard([&]() { delete handle; }); }

const char *params_last_error() { return last_error; }

void params_clear_all(ParamsHandle *handle, unsigned int flag) {
  guard([&]() { handle->params.clearAll(static_cast<ParamKeyFlag>(flag)); });
}

bool params_check_key(ParamsHandle *handle, const char *key) {
  return guard([&]() { return handle->params.checkKey(key); }, false);
}

int params_get_key_type(ParamsHandle *handle, const char *key) {
  return guard([&]() { return static_cast<int>(handle->params.getKeyType(key)); }, -1);
}

ParamsBuffer params_get_default(ParamsHandle *handle, const char *key) {
  return guard([&]() {
    auto value = handle->params.getKeyDefaultValue(key);
    return value.has_value() ? return_string(*value) : ParamsBuffer{nullptr, 0};
  }, ParamsBuffer{nullptr, 0});
}

ParamsBuffer params_get(ParamsHandle *handle, const char *key, bool block) {
  return guard([&]() { return return_string(handle->params.get(key, block)); }, ParamsBuffer{nullptr, 0});
}

bool params_get_bool(ParamsHandle *handle, const char *key, bool block) {
  return guard([&]() { return handle->params.getBool(key, block); }, false);
}

int params_put(ParamsHandle *handle, const char *key, const char *value, size_t size, bool block) {
  return guard([&]() {
    if (block) return handle->params.put(key, value, size);
    handle->params.putNonBlocking(key, std::string(value, size));
    return 0;
  }, -1);
}

int params_put_bool(ParamsHandle *handle, const char *key, bool value, bool block) {
  return guard([&]() {
    if (block) return handle->params.putBool(key, value);
    handle->params.putBoolNonBlocking(key, value);
    return 0;
  }, -1);
}

int params_remove(ParamsHandle *handle, const char *key) {
  return guard([&]() { return handle->params.remove(key); }, -1);
}

ParamsBuffer params_get_path(ParamsHandle *handle, const char *key, size_t key_size) {
  return guard([&]() { return return_string(handle->params.getParamPath(std::string(key, key_size))); }, ParamsBuffer{nullptr, 0});
}

size_t params_keys_size(ParamsHandle *handle) {
  return guard([&]() { return handle->keys.size(); }, size_t{0});
}

ParamsBuffer params_key_at(ParamsHandle *handle, size_t index) {
  return guard([&]() {
    return index < handle->keys.size() ? return_string(handle->keys[index]) : ParamsBuffer{nullptr, 0};
  }, ParamsBuffer{nullptr, 0});
}

}  // extern "C"
