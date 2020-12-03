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
    CLQueuedKernel(Thneed *lthneed,
                   cl_kernel _kernel,
                   cl_uint _work_dim,
                   const size_t *_global_work_size,
                   const size_t *_local_work_size);
    int exec(bool recreate_kernel);
    void debug_print(bool verbose);
    int get_arg_num(const char *search_arg_name);
    cl_program program;
    string name;
    cl_uint num_args;
    vector<string> arg_names;
    vector<string> args;
  private:
    Thneed *thneed;
    cl_kernel kernel;
    cl_uint work_dim;
    size_t global_work_size[3];
    size_t local_work_size[3];
};

class CachedCommand {
  public:
    CachedCommand(Thneed *lthneed, struct kgsl_gpu_command *cmd);
    void exec(bool wait);
    void disassemble();
  private:
    struct kgsl_gpu_command cache;
    struct kgsl_command_object cmds[2];
    struct kgsl_command_object objs[1];
    Thneed *thneed;
    vector<shared_ptr<CLQueuedKernel> > kq;
};

class Thneed {
  public:
    Thneed();
    void stop();
    void execute(float **finputs, float *foutput, bool slow=false);
    int optimize();

    vector<cl_mem> inputs;
    cl_mem output;

    cl_context context;
    cl_command_queue command_queue;
    int context_id;

    // protected?
    int record;
    int timestamp;
    unique_ptr<GPUMalloc> ram;
    vector<unique_ptr<CachedCommand> > cmds;
    vector<string> syncobjs;
    int fd;

    // all CL kernels
    vector<shared_ptr<CLQueuedKernel> > kq;

    // current
    vector<shared_ptr<CLQueuedKernel> > ckq;
};

