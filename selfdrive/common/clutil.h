#ifndef CLUTIL_H
#define CLUTIL_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

void clu_init(void);

cl_device_id cl_get_device_id(cl_device_type device_type);
cl_program cl_create_program_from_file(cl_context ctx, const char* path);
void cl_print_info(cl_platform_id platform, cl_device_id device);
void cl_print_build_errors(cl_program program, cl_device_id device);
void cl_print_build_errors(cl_program program, cl_device_id device);

cl_program cl_cached_program_from_hash(cl_context ctx, cl_device_id device_id, uint64_t hash);
cl_program cl_cached_program_from_string(cl_context ctx, cl_device_id device_id,
                                         const char* src, const char* args,
                                         uint64_t *out_hash);
cl_program cl_cached_program_from_file(cl_context ctx, cl_device_id device_id, const char* path, const char* args,
                                       uint64_t *out_hash);

cl_program cl_program_from_index(cl_context ctx, cl_device_id device_id, uint64_t index_hash);

cl_program cl_index_program_from_string(cl_context ctx, cl_device_id device_id,
                                        const char* src, const char* args,
                                        uint64_t index_hash);
cl_program cl_index_program_from_file(cl_context ctx, cl_device_id device_id, const char* path, const char* args,
                                      uint64_t index_hash);

uint64_t clu_index_hash(const char *s);
uint64_t clu_fnv_hash(const uint8_t *data, size_t len);

const char* cl_get_error_string(int err);

static inline int cl_check_error(int err) {
  if (err != 0) {
    fprintf(stderr, "%s\n", cl_get_error_string(err));
    exit(1);
  }
  return err;
}


// // string hash macro. compiler, I'm so sorry.
#define CLU_H1(s,i,x)   (x*65599ULL+(uint8_t)s[(i)<strlen(s)?strlen(s)-1-(i):strlen(s)])
#define CLU_H4(s,i,x)   CLU_H1(s,i,CLU_H1(s,i+1,CLU_H1(s,i+2,CLU_H1(s,i+3,x))))
#define CLU_H16(s,i,x)  CLU_H4(s,i,CLU_H4(s,i+4,CLU_H4(s,i+8,CLU_H4(s,i+12,x))))
#define CLU_H64(s,i,x)  CLU_H16(s,i,CLU_H16(s,i+16,CLU_H16(s,i+32,CLU_H16(s,i+48,x))))
// #define CLU_H256(s,i,x) CLU_H64(s,i,CLU_H64(s,i+64,CLU_H64(s,i+128,CLU_H64(s,i+192,x))))
#define CLU_H128(s,i,x) CLU_H64(s,i,CLU_H64(s,i+64,x))
#define CLU_HASH(s)    ((uint64_t)(CLU_H128(s,0,0)^(CLU_H128(s,0,0)>>32)))

#define CLU_STRINGIFY(x) #x
#define CLU_STRINGIFY2(x) CLU_STRINGIFY(x)
#define CLU_LINESTR CLU_STRINGIFY2(__LINE__)

#ifdef CLU_NO_SRC

 #define CLU_LOAD_FROM_STRING(ctx, device_id, src, args) \
  cl_program_from_index(ctx, device_id, CLU_HASH("\1" __FILE__ "\1" CLU_LINESTR) ^ clu_fnv_hash((const uint8_t*)__func__, strlen(__func__)) ^ clu_fnv_hash((const uint8_t*)args, strlen(args)))
 #define CLU_LOAD_FROM_FILE(ctx, device_id, path, args) \
  cl_program_from_index(ctx, device_id, CLU_HASH("\2" path) ^ clu_fnv_hash((const uint8_t*)args, strlen(args)))

#else

 #define CLU_LOAD_FROM_STRING(ctx, device_id, src, args) \
  cl_index_program_from_string(ctx, device_id, src, args, clu_index_hash("\1" __FILE__ "\1" CLU_LINESTR) ^ clu_fnv_hash((const uint8_t*)__func__, strlen(__func__)) ^ clu_fnv_hash((const uint8_t*)args, strlen(args)))
 #define CLU_LOAD_FROM_FILE(ctx, device_id, path, args) \
  cl_index_program_from_file(ctx, device_id, path, args, clu_index_hash("\2" path) ^ clu_fnv_hash((const uint8_t*)args, strlen(args)))

#endif

#ifdef __cplusplus
}
#endif

#endif
