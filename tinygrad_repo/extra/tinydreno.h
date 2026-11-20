#ifndef _TINYDRENO_H
#define _TINYDRENO_H

#include <stdint.h>
#include <stddef.h>

typedef void * cl_llvm_instance;

cl_llvm_instance cl_compiler_create_llvm_instance(void);
void cl_compiler_destroy_llvm_instance(cl_llvm_instance inst);

enum cl_handle_type {
  CL_HANDLE_COMPILED = 1,
  CL_HANDLE_LIBRARY,
  CL_HANDLE_LINKED
};

// handle->data for CL_HANDLE_COMPILED and CL_HANDLE_LIBRARY
struct cl_compiled_data {
  uint64_t chip_id;
  uint32_t mode;
  void    *llvm_bitcode;
  uint64_t llvm_bitcode_size;
  char    *build_log;
  uint32_t build_log_len;
  uint32_t error_code;
};

// handle->data for CL_HANDLE_LINKED
struct cl_executable_data {
  int32_t  num_kernels;
  void    *kernel_props;
  uint32_t error_code;
  char    *build_log;
  char     _unk0[0x20];
  uint64_t chip_id;
  uint32_t mode;
};

typedef struct {
  enum cl_handle_type type;
  union {
    struct cl_compiled_data *compiled;
    struct cl_executable_data *executable;
  };
} cl_handle;


#define CL_MODE_32BIT 0
#define CL_MODE_64BIT 1

#define CL_SRC_STR  0
#define CL_SRC_BLOB 1

cl_handle *cl_compiler_compile_source(cl_llvm_instance inst, uint64_t chip_id, int mode, const char *options, int p5, uint64_t p6, uint64_t p7,
                                      const char *source, uint64_t source_len, uint64_t source_type, void *p11);

cl_handle *cl_compiler_link_program(cl_llvm_instance inst, uint64_t chip_id, int mode, const char *options, int num_handles,
                                    cl_handle **input_handles);


void cl_compiler_handle_create_binary(cl_handle *handle, void **out_ptr, size_t *out_size);

// lib binary format (output of handle_create_binary for type 3)
// layout: cl_lib_header, then cl_lib_section[num_sections], then data

#define CL_LIB_PROGRAM     0
#define CL_LIB_CONSTS      6
#define CL_LIB_IMAGE       7
#define CL_LIB_CODE       10
#define CL_LIB_IMAGE_DESC 11

typedef struct {
  uint32_t id;
  uint32_t offset;
  uint32_t size;
  uint32_t count;
  uint32_t entry_size;
} cl_lib_section;

typedef struct {
  uint32_t _unk0[6];
  uint32_t num_sections;
  uint32_t _unk1[5];
  cl_lib_section sections[];
} cl_lib_header;

// at sections[CL_LIB_PROGRAM].offset
typedef struct {
  char     name[8];
  uint32_t _unk0[3];
  uint32_t fregs;
  uint32_t hregs;
} cl_lib_prog;

// at sections[CL_LIB_IMAGE_DESC].offset
typedef struct {
  char     _unk0[0xc4];
  uint32_t prg_offset;
  uint32_t pvtmem;
  char     _unk1[0x0c];
  uint32_t shmem;
  uint32_t samp_cnt;
  char     _unk2[0x28];
  uint32_t brnchstck;
  char     _unk4[0x4c];
  char     kernel_name[];
} cl_lib_img_desc;

void cl_compiler_free_handle(cl_handle *handle);
void cl_compiler_free_assembly(void *ptr);

#endif
