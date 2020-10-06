#pragma once

#include <stdint.h>
#include "include/msm_kgsl.h"
#include <vector>
#include <CL/cl.h>

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
};

class Thneed {
  public:
    Thneed();
    void stop();
    void execute(float **finputs, float *foutput, bool slow=false);

    vector<cl_mem> inputs;
    cl_mem output;

    cl_command_queue command_queue;
    int context_id;

    // protected?
    int record;
    int timestamp;
    unique_ptr<GPUMalloc> ram;
    vector<unique_ptr<CachedCommand> > cmds;
    vector<string> syncobjs;
    int fd;
};

