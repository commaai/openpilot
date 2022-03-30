#include <map>
#include <string>
#include <string.h>
#include <assert.h>
#include "thneed.h"

#include "selfdrive/common/util.h"
#include "selfdrive/common/clutil.h"

extern map<cl_program, string> g_program_source;

static int is_same_size_image(cl_mem a, cl_mem b) {
  size_t a_width, a_height, a_depth, a_array_size, a_row_pitch, a_slice_pitch;
  clGetImageInfo(a, CL_IMAGE_WIDTH, sizeof(a_width), &a_width, NULL);
  clGetImageInfo(a, CL_IMAGE_HEIGHT, sizeof(a_height), &a_height, NULL);
  clGetImageInfo(a, CL_IMAGE_DEPTH, sizeof(a_depth), &a_depth, NULL);
  clGetImageInfo(a, CL_IMAGE_ARRAY_SIZE, sizeof(a_array_size), &a_array_size, NULL);
  clGetImageInfo(a, CL_IMAGE_ROW_PITCH, sizeof(a_row_pitch), &a_row_pitch, NULL);
  clGetImageInfo(a, CL_IMAGE_SLICE_PITCH, sizeof(a_slice_pitch), &a_slice_pitch, NULL);

  size_t b_width, b_height, b_depth, b_array_size, b_row_pitch, b_slice_pitch;
  clGetImageInfo(b, CL_IMAGE_WIDTH, sizeof(b_width), &b_width, NULL);
  clGetImageInfo(b, CL_IMAGE_HEIGHT, sizeof(b_height), &b_height, NULL);
  clGetImageInfo(b, CL_IMAGE_DEPTH, sizeof(b_depth), &b_depth, NULL);
  clGetImageInfo(b, CL_IMAGE_ARRAY_SIZE, sizeof(b_array_size), &b_array_size, NULL);
  clGetImageInfo(b, CL_IMAGE_ROW_PITCH, sizeof(b_row_pitch), &b_row_pitch, NULL);
  clGetImageInfo(b, CL_IMAGE_SLICE_PITCH, sizeof(b_slice_pitch), &b_slice_pitch, NULL);

  return (a_width == b_width) && (a_height == b_height) &&
    (a_depth == b_depth) && (a_array_size == b_array_size) &&
    (a_row_pitch == b_row_pitch) && (a_slice_pitch == b_slice_pitch);
}

static cl_mem make_image_like(cl_context context, cl_mem val) {
  cl_image_format format;
  size_t width, height, row_pitch;
  clGetImageInfo(val, CL_IMAGE_FORMAT, sizeof(format), &format, NULL);
  assert(format.image_channel_order == CL_RGBA);
  assert(format.image_channel_data_type == CL_HALF_FLOAT);
  clGetImageInfo(val, CL_IMAGE_WIDTH, sizeof(width), &width, NULL);
  clGetImageInfo(val, CL_IMAGE_HEIGHT, sizeof(height), &height, NULL);
  clGetImageInfo(val, CL_IMAGE_ROW_PITCH, sizeof(row_pitch), &row_pitch, NULL);

  cl_image_desc desc = {0};
  desc.image_type = CL_MEM_OBJECT_IMAGE2D;
  desc.image_width = width;
  desc.image_height = height;
  desc.image_row_pitch = row_pitch;

  cl_mem buf = clCreateBuffer(context, CL_MEM_READ_WRITE, row_pitch*height, NULL, NULL);
  assert(buf != NULL);
  desc.buffer = buf;

  cl_int err;
  cl_mem tmp = clCreateImage(context, CL_MEM_READ_WRITE, &format, &desc, NULL, &err);
  //printf("got %d for image %zux%zu %zu\n", err, width, height, row_pitch);
  assert(tmp != NULL);

  return tmp;
}

// convolution_horizontal_reduced_reads_1x1 is 66% of the model runtime
// make that faster and the model gets faster

// this cuts ~2 ms off the model runtime right now
int Thneed::optimize() {
  const char *kernel_path = getenv("KERNEL_PATH");
  if (!kernel_path) { kernel_path = "/data/openpilot/selfdrive/modeld/thneed/kernels"; printf("no KERNEL_PATH set, defaulting to %s\n", kernel_path); }

  string convolution_;
  {
    char fn[0x100];
    snprintf(fn, sizeof(fn), "%s/%s.cl", kernel_path, "convolution_");
    convolution_ = util::read_file(fn);
  }

  // load custom kernels
  map<string, cl_program> g_programs;
  for (auto &k : kq) {
    // replace program?
    if (g_programs.find(k->name) == g_programs.end()) {
      char fn[0x100];
      snprintf(fn, sizeof(fn), "%s/%s.cl", kernel_path, k->name.c_str());
      if (util::file_exists(fn)) {
        string kernel_src = util::read_file(fn);
        if (k->name.rfind("convolution_", 0) == 0) {
          kernel_src += convolution_;
        }
        printf("building kernel %s with len %lu\n", k->name.c_str(), kernel_src.length());
        k->program = cl_program_from_source(context, device_id, kernel_src);

        // save in cache
        g_programs[k->name] = k->program;
        g_program_source[k->program] = kernel_src;
      } else {
        g_programs[k->name] = NULL;
      }
    } else {
      // cached replacement
      if (g_programs[k->name] != NULL) {
        k->program = g_programs[k->name];
      }
    }

    // hack in accumulator to convolution_horizontal_reduced_reads_1x1
    if (k->name == "convolution_horizontal_reduced_reads_1x1") {
      k->arg_names.push_back("doAccumulate");
      short doAccumulate = 0;
      k->args.push_back(string((char *)&doAccumulate, sizeof(doAccumulate)));
      k->args_size.push_back(2);
      k->arg_names.push_back("accumulator");
      k->args.push_back(k->args[k->get_arg_num("output")]);
      k->args_size.push_back(8);
      k->num_args += 2;
    }

    // assert that parameters + batchNormBiases are not used
    // since they aren't supported in custom replacement kernels
    if (k->name == "convolution_horizontal_reduced_reads_1x1" ||
        k->name == "convolution_horizontal_reduced_reads" ||
        k->name == "convolution_horizontal_reduced_reads_5_outputs") {
      string p1 = k->args[k->get_arg_num("parameters")];
      string p2 = k->args[k->get_arg_num("batchNormBiases")];
      assert(p1.length() == 8 && *((uint64_t*)p1.data()) == 0);
      assert(p2.length() == 8 && *((uint64_t*)p2.data()) == 0);
    }
  }

  // optimizer
  size_t start_size;
  do {
    start_size = kq.size();

    // get optimizations
    map<string, string> replacements;
    for (int i = 0; i < kq.size(); i++) {
      // fusing elementwise_sum + activate_image will save 3 enqueues

      // delete useless copy layers
      // saves ~0.7 ms
      if (kq[i]->name == "concatenation" || kq[i]->name == "flatten") {
        string in = kq[i]->args[kq[i]->get_arg_num("input")];
        string out = kq[i]->args[kq[i]->get_arg_num("output")];
        if (is_same_size_image(*(cl_mem*)in.data(), *(cl_mem*)out.data())) {
          cl_mem tmp = make_image_like(context, *(cl_mem *)in.data());
          replacements[in] = string((char *)&tmp, sizeof(tmp));
          replacements[out] = string((char *)&tmp, sizeof(tmp));

          kq.erase(kq.begin()+i); --i;
        }
      }

      // NOTE: if activations/accumulation are done in the wrong order, this will be wrong

      // fuse activations into convs and fc_Wtx
      // saves ~1.5 ms
      // NOTE: this changes the outputs because of rounding, should be better now!
      if (i != 0 && kq[i]->name == "activate_image") {
        if (kq[i-1]->name == "convolution_horizontal_reduced_reads_1x1" ||
            kq[i-1]->name == "convolution_horizontal_reduced_reads_5_outputs" ||
            kq[i-1]->name == "convolution_horizontal_reduced_reads" ||
            kq[i-1]->name == "convolution_horizontal_reduced_reads_depthwise" ||
            kq[i-1]->name == "convolution_horizontal_reduced_reads_depthwise_stride_1" ||
            kq[i-1]->name == "fc_Wtx") {
          string lastout = kq[i-1]->args[kq[i-1]->get_arg_num("output")];
          string in = kq[i]->args[kq[i]->get_arg_num("input")];
          string out = kq[i]->args[kq[i]->get_arg_num("output")];

          if (lastout == in) {
            short neuron = *(int*)kq[i]->args[kq[i]->get_arg_num("neuron")].data();
            assert(neuron <= 5);

            // ELU isn't supported in fc_Wtx
            assert(!(kq[i-1]->name == "fc_Wtx" && neuron == 5));

            kq[i-1]->args[kq[i-1]->get_arg_num("neuron")] = string((char *)&neuron, sizeof(neuron));

            cl_mem tmp = make_image_like(context, *(cl_mem *)lastout.data());
            replacements[in] = string((char *)&tmp, sizeof(tmp));
            replacements[out] = string((char *)&tmp, sizeof(tmp));

            kq.erase(kq.begin()+i); --i;
          }
        }
      }

      // fuse accumulation into convs and fc_Wtx
      if (i != 0 && kq[i]->name == "elementwise_sum") {
        if (kq[i-1]->name == "convolution_horizontal_reduced_reads_1x1" ||
            kq[i-1]->name == "fc_Wtx") {
          string lastout = kq[i-1]->args[kq[i-1]->get_arg_num("output")];
          string a = kq[i]->args[kq[i]->get_arg_num("a")];
          string b = kq[i]->args[kq[i]->get_arg_num("b")];
          string out = kq[i]->args[kq[i]->get_arg_num("output")];

          if (lastout == a) {
            kq[i-1]->args[kq[i-1]->get_arg_num("accumulator")] = b;
          } else if (lastout == b) {
            kq[i-1]->args[kq[i-1]->get_arg_num("accumulator")] = a;
          } else {
            continue;
          }

          cl_mem tmp = make_image_like(context, *(cl_mem *)lastout.data());
          replacements[lastout] = string((char *)&tmp, sizeof(tmp));
          replacements[out] = string((char *)&tmp, sizeof(tmp));

          short doAccumulate = 1;
          kq[i-1]->args[kq[i-1]->get_arg_num("doAccumulate")] = string((char *)&doAccumulate, sizeof(doAccumulate));

          kq.erase(kq.begin()+i); --i;
        }
      }
    }

    // remap inputs and outputs, and clear the kernels
    for (int i = 0; i < kq.size(); i++) {
      kq[i]->kernel = NULL;
      for (int j = 0; j < kq[i]->num_args; j++) {
        if (replacements.find(kq[i]->args[j]) != replacements.end()) {
          kq[i]->args[j] = replacements[kq[i]->args[j]];
        }
      }
    }

    printf("optimize %lu -> %lu\n", start_size, kq.size());
  } while (kq.size() != start_size);

  size_t work_group_size = 0;
  clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(work_group_size), &work_group_size, NULL);
  printf("max work group size %lu\n", work_group_size);

  // local work group optimizer
  for (auto &k : kq) {
    // only do it for convs, since others might share memory
    if (k->name.rfind("convolution_", 0) == 0) {
      int best = -1;
      if (k->local_work_size[0] * k->local_work_size[1] * k->local_work_size[2] < work_group_size/2) {
        uint64_t base_time = k->benchmark();
        uint64_t best_time = base_time;
        for (int i = 0; i < 3; i++) {
          k->local_work_size[i] *= 2;
          uint64_t this_time = k->benchmark();
          if (this_time < best_time) {
            best = i;
            best_time = this_time;
          }
          k->local_work_size[i] /= 2;
        }
        if (best != -1) {
          k->local_work_size[best] *= 2;
          //printf("%s %.2f ms doubled %d to %.2f ms\n", k->name.c_str(), base_time/1e6, best, best_time/1e6);
        }
      }

    }
  }

  return 0;
}

