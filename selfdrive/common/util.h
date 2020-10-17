#ifndef COMMON_UTIL_H
#define COMMON_UTIL_H

#include <stdio.h>

#ifndef sighandler_t
typedef void (*sighandler_t)(int sig);
#endif

#ifndef __cplusplus

#define min(a,b) \
 ({ __typeof__ (a) _a = (a); \
     __typeof__ (b) _b = (b); \
   _a < _b ? _a : _b; })

#define max(a,b) \
 ({ __typeof__ (a) _a = (a); \
     __typeof__ (b) _b = (b); \
   _a > _b ? _a : _b; })

#endif

#define ARRAYSIZE(x) (sizeof(x)/sizeof(x[0]))

#undef ALIGN
#define ALIGN(x, align) (((x) + (align)-1) & ~((align)-1))

#ifdef __cplusplus
extern "C" {
#endif

// Reads a file into a newly allocated buffer.
//
// Returns NULL on failure, otherwise the NULL-terminated file contents.
// The result must be freed by the caller.
void* read_file(const char* path, size_t* out_len);
int write_file(const char* path, const void* data, size_t size);

void set_thread_name(const char* name);

int set_realtime_priority(int level);
int set_core_affinity(int core);

#ifdef __cplusplus
}
#endif

#endif
