#include <cstdlib>
#include <cstring>
#include "common/params.h"

extern "C" {

void* params_create(const char* path) {
  if (path) {
    return new Params(std::string(path));
  }
  return new Params();
}

void params_destroy(void* p) {
  delete (Params*)p;
}

char* params_get(void* p, const char* key, bool block, size_t* out_len) {
  Params* params = (Params*)p;
  std::string val = params->get(key, block);
  if (val.empty()) {
    return NULL;
  }

  if (out_len) {
    *out_len = val.size();
  }

  char* ret = (char*)malloc(val.size() + 1);
  memcpy(ret, val.data(), val.size());
  ret[val.size()] = 0; // Null-terminate for convenience if used as string
  return ret;
}

// Helper to free strings returned by params_get
void params_free_str(const char* str) {
  if (str) free((void*)str);
}

int params_put(void* p, const char* key, const char* val, size_t val_size) {
  Params* params = (Params*)p;
  return params->put(key, val, val_size);
}

void params_put_nonblocking(void* p, const char* key, const char* val, size_t val_size) {
  Params* params = (Params*)p;
  params->putNonBlocking(key, std::string(val, val_size));
}

int params_put_bool(void* p, const char* key, bool val) {
  Params* params = (Params*)p;
  return params->putBool(key, val);
}

int params_remove(void* p, const char* key) {
  Params* params = (Params*)p;
  return params->remove(key);
}

void params_clear_all(void* p, int type) {
  Params* params = (Params*)p;
  params->clearAll((ParamKeyFlag)type);
}

void params_put_bool_nonblocking(void* p, const char* key, bool val) {
  Params* params = (Params*)p;
  params->putBoolNonBlocking(key, val);
}

char* params_get_param_path(void* p, const char* key) {
  Params* params = (Params*)p;
  std::string path = params->getParamPath(key ? key : "");
  return strdup(path.c_str());
}

// all_keys returns a NULL-terminated array of strings.
// The caller must free each string and the array itself.
char** params_all_keys(void* p, size_t* out_len) {
  Params* params = (Params*)p;
  std::vector<std::string> keys = params->allKeys();
  *out_len = keys.size();

  char** ret = (char**)malloc(sizeof(char*) * keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    ret[i] = strdup(keys[i].c_str());
  }
  return ret;
}

// Helper to free the array of strings returned by params_all_keys
void params_free_str_array(char** arr, size_t len) {
  if (!arr) return;
  for (size_t i = 0; i < len; ++i) {
    free(arr[i]);
  }
  free(arr);
}

char* params_get_default_value(void* p, const char* key) {
  Params* params = (Params*)p;
  std::optional<std::string> val = params->getKeyDefaultValue(key);
  if (val.has_value()) {
    return strdup(val->c_str());
  }
  return NULL;
}

bool params_check_key(void* p, const char* key) {
  Params* params = (Params*)p;
  return params->checkKey(key);
}

int params_get_key_flag(void* p, const char* key) {
  Params* params = (Params*)p;
  return (int)params->getKeyFlag(key);
}

int params_get_key_type(void* p, const char* key) {
  Params* params = (Params*)p;
  return (int)params->getKeyType(key);
}

}
