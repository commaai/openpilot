#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>
#include <string.h>
#include <unistd.h>
#include <inttypes.h>
#include <sys/stat.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "common/util.h"

#include "clutil.h"

typedef struct CLUProgramIndex {
  uint64_t index_hash;
  const uint8_t* bin_data;
  const uint8_t* bin_end;
} CLUProgramIndex;

#ifdef CLU_NO_SRC
#include "clcache_bins.h"
#else
static const CLUProgramIndex clu_index[] = {};
#endif

void clu_init(void) {
#ifndef CLU_NO_SRC
  mkdir("/tmp/clcache", 0777);
  unlink("/tmp/clcache/index.cli");
#endif
}

cl_device_id cl_get_device_id(cl_device_type device_type) {
  bool opencl_platform_found = false;
  cl_device_id device_id = NULL;
  
  cl_uint num_platforms = 0;
  int err = clGetPlatformIDs(0, NULL, &num_platforms);
  assert(err == 0);
  cl_platform_id* platform_ids = malloc(sizeof(cl_platform_id) * num_platforms);
  err = clGetPlatformIDs(num_platforms, platform_ids, NULL);
  assert(err == 0);

  char cBuffer[1024];
  for (size_t i = 0; i < num_platforms; i++) {
    err = clGetPlatformInfo(platform_ids[i], CL_PLATFORM_NAME, sizeof(cBuffer), &cBuffer, NULL);
    assert(err == 0);
    printf("platform[%zu] CL_PLATFORM_NAME: %s\n", i, cBuffer);

    cl_uint num_devices;
    err = clGetDeviceIDs(platform_ids[i], device_type, 0, NULL, &num_devices);
    if (err != 0 || !num_devices) {
      continue;
    }
    // Get first device
    err = clGetDeviceIDs(platform_ids[i], device_type, 1, &device_id, NULL);
    assert(err == 0);
    cl_print_info(platform_ids[i], device_id);
    opencl_platform_found = true;
    break;
  }
  free(platform_ids);
  
  if (!opencl_platform_found) {
    printf("No valid openCL platform found\n");
    assert(opencl_platform_found);
  }
  return device_id;
}

cl_program cl_create_program_from_file(cl_context ctx, const char* path) {
  char* src_buf = read_file(path, NULL);
  assert(src_buf);

  int err = 0;
  cl_program ret = clCreateProgramWithSource(ctx, 1, (const char**)&src_buf, NULL, &err);
  assert(err == 0);

  free(src_buf);

  return ret;
}

static char* get_version_string(cl_platform_id platform) {
  size_t size = 0;
  int err;
  err = clGetPlatformInfo(platform, CL_PLATFORM_VERSION, 0, NULL, &size);
  assert(err == 0);
  char *str = malloc(size);
  assert(str);
  err = clGetPlatformInfo(platform, CL_PLATFORM_VERSION, size, str, NULL);
  assert(err == 0);
  return str;
}

void cl_print_info(cl_platform_id platform, cl_device_id device) {
  char str[4096];

  clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, sizeof(str), str, NULL);
  printf("vendor: '%s'\n", str);

  char* version = get_version_string(platform);
  printf("platform version: '%s'\n", version);
  free(version);

  clGetPlatformInfo(platform, CL_PLATFORM_PROFILE, sizeof(str), str, NULL);
  printf("profile: '%s'\n", str);

  clGetPlatformInfo(platform, CL_PLATFORM_EXTENSIONS, sizeof(str), str, NULL);
  printf("extensions: '%s'\n", str);

  clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(str), str, NULL);
  printf("name: '%s'\n", str);

  clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(str), str, NULL);
  printf("device version: '%s'\n", str);

  size_t sz;
  clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(sz), &sz, NULL);
  printf("max work group size: %zu\n", sz);

  cl_device_type type;
  clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(type), &type, NULL);
  printf("type = 0x%04x = ", (unsigned int)type);
  switch(type) {
  case CL_DEVICE_TYPE_CPU:
    printf("CL_DEVICE_TYPE_CPU\n");
    break;
  case CL_DEVICE_TYPE_GPU:
    printf("CL_DEVICE_TYPE_GPU\n");
    break;
  case CL_DEVICE_TYPE_ACCELERATOR:
    printf("CL_DEVICE_TYPE_ACCELERATOR\n");
    break;
  default:
    printf("Other...\n" );
    break;
  }
}

void cl_print_build_errors(cl_program program, cl_device_id device) {
  cl_build_status status;
  clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_STATUS,
          sizeof(cl_build_status), &status, NULL);

  size_t log_size;
  clGetProgramBuildInfo(program, device,
          CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

  char* log = calloc(log_size+1, 1);
  assert(log);

  clGetProgramBuildInfo(program, device,
          CL_PROGRAM_BUILD_LOG, log_size+1, log, NULL);

  printf("build failed; status=%d, log:\n%s\n",
          status, log);

  free(log);
}

uint64_t clu_index_hash(const char* s) {
  size_t sl = strlen(s);
  assert(sl < 128);
  uint64_t x = 0;
  for (int i=127; i>=0; i--) {
    x *= 65599ULL;
    x += (uint8_t)s[i<sl ? sl-1-i : sl];
  }
  return x ^ (x >> 32);
}

uint64_t clu_fnv_hash(const uint8_t *data, size_t len) {
  /* 64 bit Fowler/Noll/Vo FNV-1a hash code */
  uint64_t hval = 0xcbf29ce484222325ULL;
  const uint8_t *dp = data;
  const uint8_t *de = data + len;
  while (dp < de) {
    hval ^= (uint64_t) *dp++;
    hval += (hval << 1) + (hval << 4) + (hval << 5) +
        (hval << 7) + (hval << 8) + (hval << 40);
  }

  return hval;
}

cl_program cl_cached_program_from_hash(cl_context ctx, cl_device_id device_id, uint64_t hash) {
  int err;

  char cache_path[1024];
  snprintf(cache_path, sizeof(cache_path), "/tmp/clcache/%016" PRIx64 ".clb", hash);

  size_t bin_size;
  uint8_t *bin = read_file(cache_path, &bin_size);
  if (!bin) {
    return NULL;
  }

  cl_program prg = clCreateProgramWithBinary(ctx, 1, &device_id, &bin_size, (const uint8_t**)&bin, NULL, &err);
  assert(err == 0);

  free(bin);

  err = clBuildProgram(prg, 1, &device_id, NULL, NULL, NULL);
  assert(err == 0);

  return prg;
}

#ifndef CLU_NO_CACHE
static uint8_t* get_program_binary(cl_program prg, size_t *out_size) {
  int err;

  cl_uint num_devices;
  err = clGetProgramInfo(prg, CL_PROGRAM_NUM_DEVICES, sizeof(num_devices), &num_devices, NULL);
  assert(err == 0);
  assert(num_devices == 1);

  size_t binary_size = 0;
  err = clGetProgramInfo(prg, CL_PROGRAM_BINARY_SIZES, sizeof(binary_size), &binary_size, NULL);
  assert(err == 0);
  assert(binary_size > 0);

  uint8_t *binary_buf = malloc(binary_size);
  assert(binary_buf);

  uint8_t* bufs[1] = { binary_buf, };

  err = clGetProgramInfo(prg, CL_PROGRAM_BINARIES, sizeof(bufs), &bufs, NULL);
  assert(err == 0);

  *out_size = binary_size;
  return binary_buf;
}
#endif

cl_program cl_cached_program_from_string(cl_context ctx, cl_device_id device_id,
                                         const char* src, const char* args,
                                         uint64_t *out_hash) {
  int err;

  cl_platform_id platform;
  err = clGetDeviceInfo(device_id, CL_DEVICE_PLATFORM, sizeof(platform), &platform, NULL);
  assert(err == 0);

  const char* platform_version = get_version_string(platform);

  const size_t hash_len = strlen(platform_version)+1+strlen(src)+1+strlen(args)+1;
  char* hash_buf = malloc(hash_len);
  assert(hash_buf);
  memset(hash_buf, 0, hash_len);
  snprintf(hash_buf, hash_len, "%s%c%s%c%s", platform_version, 1, src, 1, args);
  free((void*)platform_version);

  uint64_t hash = clu_fnv_hash((uint8_t*)hash_buf, hash_len);
  free(hash_buf);

  cl_program prg = NULL;
#ifndef CLU_NO_CACHE
  prg = cl_cached_program_from_hash(ctx, device_id, hash);
#endif
  if (prg == NULL) {
    prg = clCreateProgramWithSource(ctx, 1, (const char**)&src, NULL, &err);
    assert(err == 0);

    err = clBuildProgram(prg, 1, &device_id, args, NULL, NULL);
    if (err != 0) {
      cl_print_build_errors(prg, device_id);
    }
    assert(err == 0);

#ifndef CLU_NO_CACHE
    // write program binary to cache

    size_t binary_size;
    uint8_t *binary_buf = get_program_binary(prg, &binary_size);

    char cache_path[1024];
    snprintf(cache_path, sizeof(cache_path), "/tmp/clcache/%016" PRIx64 ".clb", hash);
    FILE* of = fopen(cache_path, "wb");
    assert(of);
    fwrite(binary_buf, 1, binary_size, of);
    fclose(of);

    free(binary_buf);
#endif
  }

  if (out_hash) *out_hash = hash;
  return prg;
}

cl_program cl_cached_program_from_file(cl_context ctx, cl_device_id device_id, const char* path, const char* args,
                                       uint64_t *out_hash) {
  char* src_buf = read_file(path, NULL);
  assert(src_buf);
  cl_program ret = cl_cached_program_from_string(ctx, device_id, src_buf, args, out_hash);
  free(src_buf);
  return ret;
}

#ifndef CLU_NO_CACHE
static void add_index(uint64_t index_hash, uint64_t src_hash) {
  FILE *f = fopen("/tmp/clcache/index.cli", "a");
  assert(f);
  fprintf(f, "%016" PRIx64 " %016" PRIx64 "\n", index_hash, src_hash);
  fclose(f);
}
#endif

cl_program cl_program_from_index(cl_context ctx, cl_device_id device_id, uint64_t index_hash) {
  int err;

  int i;
  for (i=0; i<ARRAYSIZE(clu_index); i++) {
    if (clu_index[i].index_hash == index_hash) {
      break;
    }
  }
  if (i >= ARRAYSIZE(clu_index)) {
    assert(false);
  }

  size_t bin_size = clu_index[i].bin_end - clu_index[i].bin_data;
  const uint8_t *bin_data = clu_index[i].bin_data;

  cl_program prg = clCreateProgramWithBinary(ctx, 1, &device_id, &bin_size, (const uint8_t**)&bin_data, NULL, &err);
  assert(err == 0);

  err = clBuildProgram(prg, 1, &device_id, NULL, NULL, NULL);
  assert(err == 0);

  return prg;
}

cl_program cl_index_program_from_string(cl_context ctx, cl_device_id device_id,
                                        const char* src, const char* args,
                                        uint64_t index_hash) {
  uint64_t src_hash = 0;
  cl_program ret = cl_cached_program_from_string(ctx, device_id, src, args, &src_hash);
#ifndef CLU_NO_CACHE
  add_index(index_hash, src_hash);
#endif
  return ret;
}

cl_program cl_index_program_from_file(cl_context ctx, cl_device_id device_id, const char* path, const char* args,
                                      uint64_t index_hash) {
  uint64_t src_hash = 0;
  cl_program ret = cl_cached_program_from_file(ctx, device_id, path, args, &src_hash);
#ifndef CLU_NO_CACHE
  add_index(index_hash, src_hash);
#endif
  return ret;
}

/*
 * Given a cl code and return a string represenation
 */
const char* cl_get_error_string(int err) {
    switch (err) {
        case 0: return "CL_SUCCESS";
        case -1: return "CL_DEVICE_NOT_FOUND";
        case -2: return "CL_DEVICE_NOT_AVAILABLE";
        case -3: return "CL_COMPILER_NOT_AVAILABLE";
        case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case -5: return "CL_OUT_OF_RESOURCES";
        case -6: return "CL_OUT_OF_HOST_MEMORY";
        case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case -8: return "CL_MEM_COPY_OVERLAP";
        case -9: return "CL_IMAGE_FORMAT_MISMATCH";
        case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case -12: return "CL_MAP_FAILURE";
        case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case -15: return "CL_COMPILE_PROGRAM_FAILURE";
        case -16: return "CL_LINKER_NOT_AVAILABLE";
        case -17: return "CL_LINK_PROGRAM_FAILURE";
        case -18: return "CL_DEVICE_PARTITION_FAILED";
        case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
        case -30: return "CL_INVALID_VALUE";
        case -31: return "CL_INVALID_DEVICE_TYPE";
        case -32: return "CL_INVALID_PLATFORM";
        case -33: return "CL_INVALID_DEVICE";
        case -34: return "CL_INVALID_CONTEXT";
        case -35: return "CL_INVALID_QUEUE_PROPERTIES";
        case -36: return "CL_INVALID_COMMAND_QUEUE";
        case -37: return "CL_INVALID_HOST_PTR";
        case -38: return "CL_INVALID_MEM_OBJECT";
        case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case -40: return "CL_INVALID_IMAGE_SIZE";
        case -41: return "CL_INVALID_SAMPLER";
        case -42: return "CL_INVALID_BINARY";
        case -43: return "CL_INVALID_BUILD_OPTIONS";
        case -44: return "CL_INVALID_PROGRAM";
        case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case -46: return "CL_INVALID_KERNEL_NAME";
        case -47: return "CL_INVALID_KERNEL_DEFINITION";
        case -48: return "CL_INVALID_KERNEL";
        case -49: return "CL_INVALID_ARG_INDEX";
        case -50: return "CL_INVALID_ARG_VALUE";
        case -51: return "CL_INVALID_ARG_SIZE";
        case -52: return "CL_INVALID_KERNEL_ARGS";
        case -53: return "CL_INVALID_WORK_DIMENSION";
        case -54: return "CL_INVALID_WORK_GROUP_SIZE";
        case -55: return "CL_INVALID_WORK_ITEM_SIZE";
        case -56: return "CL_INVALID_GLOBAL_OFFSET";
        case -57: return "CL_INVALID_EVENT_WAIT_LIST";
        case -58: return "CL_INVALID_EVENT";
        case -59: return "CL_INVALID_OPERATION";
        case -60: return "CL_INVALID_GL_OBJECT";
        case -61: return "CL_INVALID_BUFFER_SIZE";
        case -62: return "CL_INVALID_MIP_LEVEL";
        case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
        case -64: return "CL_INVALID_PROPERTY";
        case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
        case -66: return "CL_INVALID_COMPILER_OPTIONS";
        case -67: return "CL_INVALID_LINKER_OPTIONS";
        case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";
        case -69: return "CL_INVALID_PIPE_SIZE";
        case -70: return "CL_INVALID_DEVICE_QUEUE";
        case -71: return "CL_INVALID_SPEC_ID";
        case -72: return "CL_MAX_SIZE_RESTRICTION_EXCEEDED";
        case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
        case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
        case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
        case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
        case -1006: return "CL_INVALID_D3D11_DEVICE_KHR";
        case -1007: return "CL_INVALID_D3D11_RESOURCE_KHR";
        case -1008: return "CL_D3D11_RESOURCE_ALREADY_ACQUIRED_KHR";
        case -1009: return "CL_D3D11_RESOURCE_NOT_ACQUIRED_KHR";
        case -1010: return "CL_INVALID_DX9_MEDIA_ADAPTER_KHR";
        case -1011: return "CL_INVALID_DX9_MEDIA_SURFACE_KHR";
        case -1012: return "CL_DX9_MEDIA_SURFACE_ALREADY_ACQUIRED_KHR";
        case -1013: return "CL_DX9_MEDIA_SURFACE_NOT_ACQUIRED_KHR";
        case -1093: return "CL_INVALID_EGL_OBJECT_KHR";
        case -1092: return "CL_EGL_RESOURCE_NOT_ACQUIRED_KHR";
        case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
        case -1057: return "CL_DEVICE_PARTITION_FAILED_EXT";
        case -1058: return "CL_INVALID_PARTITION_COUNT_EXT";
        case -1059: return "CL_INVALID_PARTITION_NAME_EXT";
        case -1094: return "CL_INVALID_ACCELERATOR_INTEL";
        case -1095: return "CL_INVALID_ACCELERATOR_TYPE_INTEL";
        case -1096: return "CL_INVALID_ACCELERATOR_DESCRIPTOR_INTEL";
        case -1097: return "CL_ACCELERATOR_TYPE_NOT_SUPPORTED_INTEL";
        case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
        case -1098: return "CL_INVALID_VA_API_MEDIA_ADAPTER_INTEL";
        case -1099: return "CL_INVALID_VA_API_MEDIA_SURFACE_INTEL";
        case -1100: return "CL_VA_API_MEDIA_SURFACE_ALREADY_ACQUIRED_INTEL";
        case -1101: return "CL_VA_API_MEDIA_SURFACE_NOT_ACQUIRED_INTEL";
        default: return "CL_UNKNOWN_ERROR";
    }
}
