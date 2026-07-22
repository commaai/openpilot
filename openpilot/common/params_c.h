#pragma once

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ParamsHandle ParamsHandle;

typedef struct {
  const char *data;
  size_t size;
} ParamsBuffer;

ParamsHandle *params_create(const char *path);
void params_destroy(ParamsHandle *handle);
const char *params_last_error(void);

void params_clear_all(ParamsHandle *handle, unsigned int flag);
bool params_check_key(ParamsHandle *handle, const char *key);
int params_get_key_type(ParamsHandle *handle, const char *key);
ParamsBuffer params_get_default(ParamsHandle *handle, const char *key);
ParamsBuffer params_get(ParamsHandle *handle, const char *key, bool block);
bool params_get_bool(ParamsHandle *handle, const char *key, bool block);
int params_put(ParamsHandle *handle, const char *key, const char *value, size_t size, bool block);
int params_put_bool(ParamsHandle *handle, const char *key, bool value, bool block);
int params_remove(ParamsHandle *handle, const char *key);
ParamsBuffer params_get_path(ParamsHandle *handle, const char *key);
size_t params_keys_size(ParamsHandle *handle);
ParamsBuffer params_key_at(ParamsHandle *handle, size_t index);

#ifdef __cplusplus
}
#endif
