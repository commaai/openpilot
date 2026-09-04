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
  ParamsHandle(const char *path, size_t path_size) : params(std::string(path, path_size)), keys(params.allKeys()) {
  }

  Params params;
  const std::vector<std::string> keys;
};

namespace {
thread_local char last_error[512] = {};
thread_local std::string result;

void set_error(const char *error) {
  snprintf(last_error, sizeof(last_error), "%s", error);
}

ParamsBuffer return_string(std::string value) {
  result = std::move(value);
  return {result.data(), result.size()};
}

template <typename Result, typename Callable>
Result translate_exceptions(Result failure, Callable &&callable) noexcept {
  last_error[0] = '\0';
  try {
    return callable();
  } catch (const std::exception &e) {
    set_error(e.what());
  } catch (...) {
    set_error("unknown C++ exception");
  }
  return failure;
}

template <typename Callable>
void translate_exceptions(Callable &&callable) noexcept {
  translate_exceptions(false, [&]() {
    callable();
    return true;
  });
}
}  // namespace

extern "C" {

ParamsHandle *params_create(const char *path, size_t path_size) noexcept {
  return translate_exceptions(static_cast<ParamsHandle *>(nullptr), [&]() {
    return new ParamsHandle(path, path_size);
  });
}

void params_destroy(ParamsHandle *handle) noexcept {
  translate_exceptions([&]() {
    delete handle;
  });
}

const char *params_last_error() noexcept {
  return last_error;
}

void params_clear_all(ParamsHandle *handle, unsigned int flag) noexcept {
  translate_exceptions([&]() {
    handle->params.clearAll(static_cast<ParamKeyFlag>(flag));
  });
}

bool params_check_key(ParamsHandle *handle, const char *key) noexcept {
  return translate_exceptions(false, [&]() {
    return handle->params.checkKey(key);
  });
}

int params_get_key_type(ParamsHandle *handle, const char *key) noexcept {
  return translate_exceptions(-1, [&]() {
    return static_cast<int>(handle->params.getKeyType(key));
  });
}

unsigned int params_get_key_flag(ParamsHandle *handle, const char *key) noexcept {
  return translate_exceptions(0U, [&]() {
    return static_cast<unsigned int>(handle->params.getKeyFlag(key));
  });
}

ParamsBuffer params_get_default(ParamsHandle *handle, const char *key) noexcept {
  return translate_exceptions(ParamsBuffer{nullptr, 0}, [&]() {
    auto value = handle->params.getKeyDefaultValue(key);
    if (!value.has_value()) {
      return ParamsBuffer{nullptr, 0};
    }
    return return_string(*value);
  });
}

ParamsBuffer params_get(ParamsHandle *handle, const char *key, bool block) noexcept {
  return translate_exceptions(ParamsBuffer{nullptr, 0}, [&]() {
    return return_string(handle->params.get(key, block));
  });
}

bool params_get_bool(ParamsHandle *handle, const char *key, bool block) noexcept {
  return translate_exceptions(false, [&]() {
    return handle->params.getBool(key, block);
  });
}

int params_put(ParamsHandle *handle, const char *key, const char *value, size_t size, bool block) noexcept {
  return translate_exceptions(-1, [&]() {
    if (block) {
      return handle->params.put(key, value, size);
    }
    handle->params.putNonBlocking(key, std::string(value, size));
    return 0;
  });
}

int params_put_bool(ParamsHandle *handle, const char *key, bool value, bool block) noexcept {
  return translate_exceptions(-1, [&]() {
    if (block) {
      return handle->params.putBool(key, value);
    }
    handle->params.putBoolNonBlocking(key, value);
    return 0;
  });
}

int params_remove(ParamsHandle *handle, const char *key) noexcept {
  return translate_exceptions(-1, [&]() {
    return handle->params.remove(key);
  });
}

ParamsBuffer params_get_path(ParamsHandle *handle, const char *key, size_t key_size) noexcept {
  return translate_exceptions(ParamsBuffer{nullptr, 0}, [&]() {
    return return_string(handle->params.getParamPath(std::string(key, key_size)));
  });
}

size_t params_keys_size(ParamsHandle *handle) noexcept {
  return translate_exceptions(size_t{0}, [&]() {
    return handle->keys.size();
  });
}

ParamsBuffer params_key_at(ParamsHandle *handle, size_t index) noexcept {
  return translate_exceptions(ParamsBuffer{nullptr, 0}, [&]() {
    if (index >= handle->keys.size()) {
      return ParamsBuffer{nullptr, 0};
    }
    return return_string(handle->keys[index]);
  });
}

}  // extern "C"
