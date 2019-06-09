#ifndef _SELFDRIVE_COMMON_PARAMS_H_
#define _SELFDRIVE_COMMON_PARAMS_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int write_db_value(const char* params_path, const char* key, const char* value,
                   size_t value_size);

// Reads a value from the params database.
// Inputs:
//  params_path: The path of the database, or NULL to use the default.
//  key: The key to read.
//  value: A pointer where a newly allocated string containing the db value will
//         be written.
//  value_sz: A pointer where the size of value will be written. Does not
//            include the NULL terminator.
//
// Returns: Negative on failure, otherwise 0.
int read_db_value(const char* params_path, const char* key, char** value,
                  size_t* value_sz);

// Reads a value from the params database, blocking until successful.
// Inputs are the same as read_db_value.
void read_db_value_blocking(const char* params_path, const char* key,
                            char** value, size_t* value_sz);

#ifdef __cplusplus
}  // extern "C"
#endif

#ifdef __cplusplus
#include <map>
#include <string>
int read_db_all(const char* params_path, std::map<std::string, std::string> *params);
#endif

#endif  // _SELFDRIVE_COMMON_PARAMS_H_
