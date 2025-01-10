#include "sunnypilot/modeld/thneed/thneed.h"

#include <cassert>
#include <cstring>
#include <map>

#include "common/clutil.h"
#include "common/timing.h"

map<pair<cl_kernel, int>, string> g_args;
map<pair<cl_kernel, int>, int> g_args_size;
map<cl_program, string> g_program_source;

void Thneed::stop() {
  //printf("Thneed::stop: recorded %lu commands\n", cmds.size());
  record = false;
}

void Thneed::clinit() {
  device_id = cl_get_device_id(CL_DEVICE_TYPE_DEFAULT);
  if (context == NULL) context = CL_CHECK_ERR(clCreateContext(NULL, 1, &device_id, NULL, NULL, &err));
  //cl_command_queue_properties props[3] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
  cl_command_queue_properties props[3] = {CL_QUEUE_PROPERTIES, 0, 0};
  command_queue = CL_CHECK_ERR(clCreateCommandQueueWithProperties(context, device_id, props, &err));
  printf("Thneed::clinit done\n");
}

cl_int Thneed::clexec() {
  if (debug >= 1) printf("Thneed::clexec: running %lu queued kernels\n", kq.size());
  for (auto &k : kq) {
    if (record) ckq.push_back(k);
    cl_int ret = k->exec();
    assert(ret == CL_SUCCESS);
  }
  return clFinish(command_queue);
}

void Thneed::copy_inputs(float **finputs, bool internal) {
  for (int idx = 0; idx < inputs.size(); ++idx) {
    if (debug >= 1) printf("copying %lu -- %p -> %p (cl %p)\n", input_sizes[idx], finputs[idx], inputs[idx], input_clmem[idx]);

    if (internal) {
      // if it's internal, using memcpy is fine since the buffer sync is cached in the ioctl layer
      if (finputs[idx] != NULL) memcpy(inputs[idx], finputs[idx], input_sizes[idx]);
    } else {
      if (finputs[idx] != NULL) CL_CHECK(clEnqueueWriteBuffer(command_queue, input_clmem[idx], CL_TRUE, 0, input_sizes[idx], finputs[idx], 0, NULL, NULL));
    }
  }
}

void Thneed::copy_output(float *foutput) {
  if (output != NULL) {
    size_t sz;
    clGetMemObjectInfo(output, CL_MEM_SIZE, sizeof(sz), &sz, NULL);
    if (debug >= 1) printf("copying %lu for output %p -> %p\n", sz, output, foutput);
    CL_CHECK(clEnqueueReadBuffer(command_queue, output, CL_TRUE, 0, sz, foutput, 0, NULL, NULL));
  } else {
    printf("CAUTION: model output is NULL, does it have no outputs?\n");
  }
}

// *********** CLQueuedKernel ***********

CLQueuedKernel::CLQueuedKernel(Thneed *lthneed,
                               cl_kernel _kernel,
                               cl_uint _work_dim,
                               const size_t *_global_work_size,
                               const size_t *_local_work_size) {
  thneed = lthneed;
  kernel = _kernel;
  work_dim = _work_dim;
  assert(work_dim <= 3);
  for (int i = 0; i < work_dim; i++) {
    global_work_size[i] = _global_work_size[i];
    local_work_size[i] = _local_work_size[i];
  }

  char _name[0x100];
  clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, sizeof(_name), _name, NULL);
  name = string(_name);
  clGetKernelInfo(kernel, CL_KERNEL_NUM_ARGS, sizeof(num_args), &num_args, NULL);

  // get args
  for (int i = 0; i < num_args; i++) {
    char arg_name[0x100] = {0};
    clGetKernelArgInfo(kernel, i, CL_KERNEL_ARG_NAME, sizeof(arg_name), arg_name, NULL);
    arg_names.push_back(string(arg_name));
    clGetKernelArgInfo(kernel, i, CL_KERNEL_ARG_TYPE_NAME, sizeof(arg_name), arg_name, NULL);
    arg_types.push_back(string(arg_name));

    args.push_back(g_args[make_pair(kernel, i)]);
    args_size.push_back(g_args_size[make_pair(kernel, i)]);
  }

  // get program
  clGetKernelInfo(kernel, CL_KERNEL_PROGRAM, sizeof(program), &program, NULL);
}

int CLQueuedKernel::get_arg_num(const char *search_arg_name) {
  for (int i = 0; i < num_args; i++) {
    if (arg_names[i] == search_arg_name) return i;
  }
  printf("failed to find %s in %s\n", search_arg_name, name.c_str());
  assert(false);
}

cl_int CLQueuedKernel::exec() {
  if (kernel == NULL) {
    kernel = clCreateKernel(program, name.c_str(), NULL);
    arg_names.clear();
    arg_types.clear();

    for (int j = 0; j < num_args; j++) {
      char arg_name[0x100] = {0};
      clGetKernelArgInfo(kernel, j, CL_KERNEL_ARG_NAME, sizeof(arg_name), arg_name, NULL);
      arg_names.push_back(string(arg_name));
      clGetKernelArgInfo(kernel, j, CL_KERNEL_ARG_TYPE_NAME, sizeof(arg_name), arg_name, NULL);
      arg_types.push_back(string(arg_name));

      cl_int ret;
      if (args[j].size() != 0) {
        assert(args[j].size() == args_size[j]);
        ret = thneed_clSetKernelArg(kernel, j, args[j].size(), args[j].data());
      } else {
        ret = thneed_clSetKernelArg(kernel, j, args_size[j], NULL);
      }
      assert(ret == CL_SUCCESS);
    }
  }

  if (thneed->debug >= 1) {
    debug_print(thneed->debug >= 2);
  }

  return clEnqueueNDRangeKernel(thneed->command_queue,
    kernel, work_dim, NULL, global_work_size, local_work_size, 0, NULL, NULL);
}

void CLQueuedKernel::debug_print(bool verbose) {
  printf("%p %56s -- ", kernel, name.c_str());
  for (int i = 0; i < work_dim; i++) {
    printf("%4zu ", global_work_size[i]);
  }
  printf(" -- ");
  for (int i = 0; i < work_dim; i++) {
    printf("%4zu ", local_work_size[i]);
  }
  printf("\n");

  if (verbose) {
    for (int i = 0; i < num_args; i++) {
      string arg = args[i];
      printf("  %s %s", arg_types[i].c_str(), arg_names[i].c_str());
      void *arg_value = (void*)arg.data();
      int arg_size = arg.size();
      if (arg_size == 0) {
        printf(" (size) %d", args_size[i]);
      } else if (arg_size == 1) {
        printf(" = %d", *((char*)arg_value));
      } else if (arg_size == 2) {
        printf(" = %d", *((short*)arg_value));
      } else if (arg_size == 4) {
        if (arg_types[i] == "float") {
          printf(" = %f", *((float*)arg_value));
        } else {
          printf(" = %d", *((int*)arg_value));
        }
      } else if (arg_size == 8) {
        cl_mem val = (cl_mem)(*((uintptr_t*)arg_value));
        printf(" = %p", val);
        if (val != NULL) {
          cl_mem_object_type obj_type;
          clGetMemObjectInfo(val, CL_MEM_TYPE, sizeof(obj_type), &obj_type, NULL);
          if (arg_types[i] == "image2d_t" || arg_types[i] == "image1d_t" || obj_type == CL_MEM_OBJECT_IMAGE2D) {
            cl_image_format format;
            size_t width, height, depth, array_size, row_pitch, slice_pitch;
            cl_mem buf;
            clGetImageInfo(val, CL_IMAGE_FORMAT, sizeof(format), &format, NULL);
            assert(format.image_channel_order == CL_RGBA);
            assert(format.image_channel_data_type == CL_HALF_FLOAT || format.image_channel_data_type == CL_FLOAT);
            clGetImageInfo(val, CL_IMAGE_WIDTH, sizeof(width), &width, NULL);
            clGetImageInfo(val, CL_IMAGE_HEIGHT, sizeof(height), &height, NULL);
            clGetImageInfo(val, CL_IMAGE_ROW_PITCH, sizeof(row_pitch), &row_pitch, NULL);
            clGetImageInfo(val, CL_IMAGE_DEPTH, sizeof(depth), &depth, NULL);
            clGetImageInfo(val, CL_IMAGE_ARRAY_SIZE, sizeof(array_size), &array_size, NULL);
            clGetImageInfo(val, CL_IMAGE_SLICE_PITCH, sizeof(slice_pitch), &slice_pitch, NULL);
            assert(depth == 0);
            assert(array_size == 0);
            assert(slice_pitch == 0);

            clGetImageInfo(val, CL_IMAGE_BUFFER, sizeof(buf), &buf, NULL);
            size_t sz = 0;
            if (buf != NULL) clGetMemObjectInfo(buf, CL_MEM_SIZE, sizeof(sz), &sz, NULL);
            printf(" image %zu x %zu rp %zu @ %p buffer %zu", width, height, row_pitch, buf, sz);
          } else {
            size_t sz;
            clGetMemObjectInfo(val, CL_MEM_SIZE, sizeof(sz), &sz, NULL);
            printf(" buffer %zu", sz);
          }
        }
      }
      printf("\n");
    }
  }
}

cl_int thneed_clSetKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void *arg_value) {
  g_args_size[make_pair(kernel, arg_index)] = arg_size;
  if (arg_value != NULL) {
    g_args[make_pair(kernel, arg_index)] = string((char*)arg_value, arg_size);
  } else {
    g_args[make_pair(kernel, arg_index)] = string("");
  }
  cl_int ret = clSetKernelArg(kernel, arg_index, arg_size, arg_value);
  return ret;
}
