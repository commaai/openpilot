#pragma once

#ifndef __user
#define __user __attribute__(())
#endif

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include <CL/cl.h>

#include "third_party/linux/include/msm_kgsl.h"

using namespace std;

cl_int thneed_clSetKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void *arg_value);

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
    vector<string> arg_types;
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

class CachedIoctl {
  public:
    virtual void exec() {}
};

class CachedSync: public CachedIoctl {
  public:
    CachedSync(Thneed *lthneed, string ldata) { thneed = lthneed; data = ldata; }
    void exec();
  private:
    Thneed *thneed;
    string data;
};

class CachedCommand: public CachedIoctl {
  public:
    CachedCommand(Thneed *lthneed, struct kgsl_gpu_command *cmd);
    void exec();
  private:
    void disassemble(int cmd_index);
    struct kgsl_gpu_command cache;
    unique_ptr<kgsl_command_object[]> cmds;
    unique_ptr<kgsl_command_object[]> objs;
    Thneed *thneed;
    vector<shared_ptr<CLQueuedKernel> > kq;
};

class Thneed {
  public:
    Thneed(bool do_clinit=false, cl_context _context = NULL);
    void stop();
    void execute(float **finputs, float *foutput, bool slow=false);
    void wait();

    vector<cl_mem> input_clmem;
    vector<void *> inputs;
    vector<size_t> input_sizes;
    cl_mem output = NULL;

    cl_context context = NULL;
    cl_command_queue command_queue;
    cl_device_id device_id;
    int context_id;

    // protected?
    bool record = false;
    int debug;
    int timestamp;

#ifdef QCOM2
    unique_ptr<GPUMalloc> ram;
    vector<unique_ptr<CachedIoctl> > cmds;
    int fd;
#endif

    // all CL kernels
    void copy_inputs(float **finputs, bool internal=false);
    void copy_output(float *foutput);
    cl_int clexec();
    vector<shared_ptr<CLQueuedKernel> > kq;

    // pending CL kernels
    vector<shared_ptr<CLQueuedKernel> > ckq;

    // loading
    void load(const char *filename);
  private:
    void clinit();
};

