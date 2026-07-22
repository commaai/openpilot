#include "common/params.h"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>

namespace {
thread_local std::string last_error;

char *copy_string(const std::string &value, size_t *size) {
  *size = value.size();
  char *result = static_cast<char *>(malloc(value.size() + 1));
  if (result != nullptr) {
    memcpy(result, value.data(), value.size());
    result[value.size()] = '\0';
  }
  return result;
}
}  // namespace

extern "C" {
void *params_create(const char *path) {
  try {
    return new Params(path);
  } catch (const std::exception &e) {
    last_error = e.what();
    return nullptr;
  }
}
void params_destroy(void *p) { delete static_cast<Params *>(p); }
const char *params_last_error() { return last_error.c_str(); }
void params_clear_all(void *p, unsigned int flag) { static_cast<Params *>(p)->clearAll(static_cast<ParamKeyFlag>(flag)); }
int params_check_key(void *p, const char *key) { return static_cast<Params *>(p)->checkKey(key); }
int params_get_key_type(void *p, const char *key) { return static_cast<Params *>(p)->getKeyType(key); }
char *params_get_default(void *p, const char *key, size_t *size, int *present) {
  auto value = static_cast<Params *>(p)->getKeyDefaultValue(key);
  *present = value.has_value();
  return value.has_value() ? copy_string(*value, size) : nullptr;
}
char *params_get(void *p, const char *key, int block, size_t *size) {
  return copy_string(static_cast<Params *>(p)->get(key, block), size);
}
int params_get_bool(void *p, const char *key, int block) { return static_cast<Params *>(p)->getBool(key, block); }
int params_put(void *p, const char *key, const char *value, size_t size, int block) {
  Params *params = static_cast<Params *>(p);
  if (block) return params->put(key, value, size);
  params->putNonBlocking(key, std::string(value, size));
  return 0;
}
int params_put_bool(void *p, const char *key, int value, int block) {
  Params *params = static_cast<Params *>(p);
  if (block) return params->putBool(key, value);
  params->putBoolNonBlocking(key, value);
  return 0;
}
int params_remove(void *p, const char *key) { return static_cast<Params *>(p)->remove(key); }
char *params_get_path(void *p, const char *key, size_t *size) {
  return copy_string(static_cast<Params *>(p)->getParamPath(key), size);
}
size_t params_keys_size(void *p) { return static_cast<Params *>(p)->allKeys().size(); }
char *params_key_at(void *p, size_t index, size_t *size) {
  return copy_string(static_cast<Params *>(p)->allKeys().at(index), size);
}
void params_free(void *value) { free(value); }
}
