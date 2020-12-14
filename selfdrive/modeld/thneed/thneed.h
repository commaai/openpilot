#pragma once

#ifndef __user
  #define __user __attribute__(())
#endif

#include <stdlib.h>
#include <stdint.h>
#include "include/msm_kgsl.h"
#include <vector>
#include <memory>
#include <string>
#include <CL/cl.h>

#define THNEED_RECORD 1
#define THNEED_DEBUG 2
#define THNEED_VERBOSE_DEBUG 4

using namespace std;

namespace json11 {
  class Json;
}
class Thneed;

class GPUMalloc {
  public:
    GPUMalloc(int size, int fd);
    ~GPUMalloc();
    void *alloc(int size);
  private:
    uint64_t base;
    int remaining;
};

class CLQueuedKernel {
  public:
    CLQueuedKernel(Thneed *lthneed) { thneed = lthneed; }
    CLQueuedKernel(Thneed *lthneed,
                   cl_kernel _kernel,
                   cl_uint _work_dim,
                   const size_t *_global_work_size,
                   const size_t *_local_work_size);
    cl_int exec();
    void debug_print(bool verbose);
    int get_arg_num(const char *search_arg_name);
    cl_program program;
    string name;
    cl_uint num_args;
    vector<string> arg_names;
    vector<string> args;
    vector<int> args_size;
    cl_kernel kernel = NULL;
    json11::Json to_json() const;

    cl_uint work_dim;
    size_t global_work_size[3] = {0};
    size_t local_work_size[3] = {0};
  private:
    Thneed *thneed;
};

class CachedCommand {
  public:
    CachedCommand(Thneed *lthneed, struct kgsl_gpu_command *cmd);
    void exec(bool wait);
    void disassemble(int cmd_index);
  private:
    struct kgsl_gpu_command cache;
    unique_ptr<kgsl_command_object[]> cmds;
    unique_ptr<kgsl_command_object[]> objs;
    Thneed *thneed;
    vector<shared_ptr<CLQueuedKernel> > kq;
};

class Thneed {
  public:
    Thneed(bool do_clinit=false);
    void stop();
    void execute(float **finputs, float *foutput, bool slow=false);
    int optimize();

    vector<cl_mem> inputs;
    cl_mem output = NULL;

    cl_context context = NULL;
    cl_command_queue command_queue;
    cl_device_id device_id;
    int context_id;

    // protected?
    int record;
    int timestamp;
    unique_ptr<GPUMalloc> ram;
    vector<unique_ptr<CachedCommand> > cmds;
    vector<string> syncobjs;
    int fd;

    // all CL kernels
    cl_int clexec();
    vector<shared_ptr<CLQueuedKernel> > kq;

    // pending CL kernels
    vector<shared_ptr<CLQueuedKernel> > ckq;

    // loading and saving
    void load(const char *filename);
    void save(const char *filename);
  private:
    void clinit();
    json11::Json to_json();
};

