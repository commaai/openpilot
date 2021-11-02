#include <cassert>
#include <set>

#include "json11.hpp"
#include "selfdrive/common/util.h"
#include "selfdrive/modeld/thneed/thneed.h"
using namespace json11;

extern map<cl_program, string> g_program_source;

void Thneed::load(const char *filename) {
  printf("Thneed::load: loading from %s\n", filename);

  string buf = util::read_file(filename);
  int jsz = *(int *)buf.data();
  string err;
  string jj(buf.data() + sizeof(int), jsz);
  Json jdat = Json::parse(jj, err);

  map<cl_mem, cl_mem> real_mem;
  real_mem[NULL] = NULL;

  int ptr = sizeof(int)+jsz;
  for (auto &obj : jdat["objects"].array_items()) {
    auto mobj = obj.object_items();
    int sz = mobj["size"].int_value();
    cl_mem clbuf = NULL;

    if (mobj["buffer_id"].string_value().size() > 0) {
      // image buffer must already be allocated
      clbuf = real_mem[*(cl_mem*)(mobj["buffer_id"].string_value().data())];
      assert(mobj["needs_load"].bool_value() == false);
    } else {
      if (mobj["needs_load"].bool_value()) {
        //printf("loading %p %d @ 0x%X\n", clbuf, sz, ptr);
        clbuf = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, sz, &buf[ptr], NULL);
        ptr += sz;
      } else {
        clbuf = clCreateBuffer(context, CL_MEM_READ_WRITE, sz, NULL, NULL);
      }
    }
    assert(clbuf != NULL);

    if (mobj["arg_type"] == "image2d_t" || mobj["arg_type"] == "image1d_t") {
      cl_image_desc desc = {0};
      desc.image_type = (mobj["arg_type"] == "image2d_t") ? CL_MEM_OBJECT_IMAGE2D : CL_MEM_OBJECT_IMAGE1D_BUFFER;
      desc.image_width = mobj["width"].int_value();
      desc.image_height = mobj["height"].int_value();
      desc.image_row_pitch = mobj["row_pitch"].int_value();
      desc.buffer = clbuf;

      cl_image_format format;
      format.image_channel_order = CL_RGBA;
      format.image_channel_data_type = CL_HALF_FLOAT;

      clbuf = clCreateImage(context, CL_MEM_READ_WRITE, &format, &desc, NULL, NULL);
      assert(clbuf != NULL);
    }

    real_mem[*(cl_mem*)(mobj["id"].string_value().data())] = clbuf;
  }

  map<string, cl_program> g_programs;
  for (auto &obj : jdat["programs"].object_items()) {
    const char *srcs[1];
    srcs[0] = (const char *)obj.second.string_value().c_str();
    size_t length = obj.second.string_value().size();

    if (record & THNEED_DEBUG) printf("building %s with size %zu\n", obj.first.c_str(), length);

    cl_program program = clCreateProgramWithSource(context, 1, srcs, &length, NULL);
    int err_ = clBuildProgram(program, 1, &device_id, "", NULL, NULL);
    if (err_ != 0) {
      printf("got err %d\n", err_);
      size_t length_;
      char buffer[2048];
      clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &length_);
      buffer[length_] = '\0';
      printf("%s\n", buffer);
    }
    assert(err_ == 0);

    g_programs[obj.first] = program;
  }

  for (auto &obj : jdat["binaries"].array_items()) {
    string name = obj["name"].string_value();
    size_t length = obj["length"].int_value();
    const unsigned char *srcs[1];
    srcs[0] = (const unsigned char *)&buf[ptr];
    ptr += length;

    if (record & THNEED_DEBUG) printf("binary %s with size %zu\n", name.c_str(), length);

    cl_int err_;
    cl_program program = clCreateProgramWithBinary(context, 1, &device_id, &length, srcs, NULL, &err_);
    assert(program != NULL && err_ == CL_SUCCESS);
    err_ = clBuildProgram(program, 1, &device_id, "", NULL, NULL);
    assert(err_ == CL_SUCCESS);

    g_programs[name] = program;
  }

  for (auto &obj : jdat["kernels"].array_items()) {
    auto gws = obj["global_work_size"];
    auto lws = obj["local_work_size"];
    auto kk = shared_ptr<CLQueuedKernel>(new CLQueuedKernel(this));

    kk->name = obj["name"].string_value();
    kk->program = g_programs[kk->name];
    kk->work_dim = obj["work_dim"].int_value();
    for (int i = 0; i < kk->work_dim; i++) {
      kk->global_work_size[i] = gws[i].int_value();
      kk->local_work_size[i] = lws[i].int_value();
    }
    kk->num_args = obj["num_args"].int_value();
    for (int i = 0; i < kk->num_args; i++) {
      string arg = obj["args"].array_items()[i].string_value();
      int arg_size = obj["args_size"].array_items()[i].int_value();
      kk->args_size.push_back(arg_size);
      if (arg_size == 8) {
        cl_mem val = *(cl_mem*)(arg.data());
        val = real_mem[val];
        kk->args.push_back(string((char*)&val, sizeof(val)));
      } else {
        kk->args.push_back(arg);
      }
    }
    kq.push_back(kk);
  }

  clFinish(command_queue);
}

void Thneed::save(const char *filename, bool save_binaries) {
  printf("Thneed::save: saving to %s\n", filename);

  // get kernels
  std::vector<Json> kernels;
  std::set<string> saved_objects;
  std::vector<Json> objects;
  std::map<string, string> programs;
  std::map<string, string> binaries;

  for (auto &k : kq) {
    kernels.push_back(k->to_json());

    // check args for objects
    int i = 0;
    for (auto &a : k->args) {
      if (a.size() == 8) {
        if (saved_objects.find(a) == saved_objects.end()) {
          saved_objects.insert(a);
          cl_mem val = *(cl_mem*)(a.data());
          if (val != NULL) {
            bool needs_load = k->arg_names[i] == "weights" || k->arg_names[i] == "biases";

            auto jj = Json::object({
              {"id", a},
              {"arg_type", k->arg_types[i]},
            });

            if (k->arg_types[i] == "image2d_t" || k->arg_types[i] == "image1d_t") {
              cl_mem buf;
              clGetImageInfo(val, CL_IMAGE_BUFFER, sizeof(buf), &buf, NULL);
              string aa = string((char *)&buf, sizeof(buf));
              jj["buffer_id"] = aa;

              size_t width, height, row_pitch;
              clGetImageInfo(val, CL_IMAGE_WIDTH, sizeof(width), &width, NULL);
              clGetImageInfo(val, CL_IMAGE_HEIGHT, sizeof(height), &height, NULL);
              clGetImageInfo(val, CL_IMAGE_ROW_PITCH, sizeof(row_pitch), &row_pitch, NULL);
              jj["width"] = (int)width;
              jj["height"] = (int)height;
              jj["row_pitch"] = (int)row_pitch;
              jj["size"] = (int)(height * row_pitch);
              jj["needs_load"] = false;

              if (saved_objects.find(aa) == saved_objects.end()) {
                saved_objects.insert(aa);
                size_t sz;
                clGetMemObjectInfo(buf, CL_MEM_SIZE, sizeof(sz), &sz, NULL);
                // save the buffer
                objects.push_back(Json::object({
                  {"id", aa},
                  {"arg_type", "<image buffer>"},
                  {"needs_load", needs_load},
                  {"size", (int)sz}
                }));
                if (needs_load) assert(sz == height * row_pitch);
              }
            } else {
              size_t sz = 0;
              clGetMemObjectInfo(val, CL_MEM_SIZE, sizeof(sz), &sz, NULL);
              jj["size"] = (int)sz;
              jj["needs_load"] = needs_load;
            }

            objects.push_back(jj);
          }
        }
      }
      i++;
    }

    if (save_binaries) {
      int err;
      size_t binary_size = 0;
      err = clGetProgramInfo(k->program, CL_PROGRAM_BINARY_SIZES, sizeof(binary_size), &binary_size, NULL);
      assert(err == 0);
      assert(binary_size > 0);
      string sv(binary_size, '\x00');

      uint8_t* bufs[1] = { (uint8_t*)sv.data(), };
      err = clGetProgramInfo(k->program, CL_PROGRAM_BINARIES, sizeof(bufs), &bufs, NULL);
      assert(err == 0);

      binaries[k->name] = sv;
    } else {
      programs[k->name] = g_program_source[k->program];
    }
  }

  vector<string> saved_buffers;
  for (auto &obj : objects) {
    auto mobj = obj.object_items();
    cl_mem val = *(cl_mem*)(mobj["id"].string_value().data());
    int sz = mobj["size"].int_value();
    if (mobj["needs_load"].bool_value()) {
      char *buf = (char *)malloc(sz);
      if (mobj["arg_type"] == "image2d_t" || mobj["arg_type"] == "image1d_t") {
        assert(false);
      } else {
        // buffers allocated with CL_MEM_HOST_WRITE_ONLY, hence this hack
        //hexdump((uint32_t*)val, 0x100);

        // the worst hack in thneed, the flags are at 0x14
        ((uint32_t*)val)[0x14] &= ~CL_MEM_HOST_WRITE_ONLY;
        cl_int ret = clEnqueueReadBuffer(command_queue, val, CL_TRUE, 0, sz, buf, 0, NULL, NULL);
        assert(ret == CL_SUCCESS);
      }
      //printf("saving buffer: %d %p %s\n", sz, buf, mobj["arg_type"].string_value().c_str());
      saved_buffers.push_back(string(buf, sz));
      free(buf);
    }
  }

  std::vector<Json> jbinaries;
  for (auto &obj : binaries) {
    jbinaries.push_back(Json::object({{"name", obj.first}, {"length", (int)obj.second.size()}}));
    saved_buffers.push_back(obj.second);
  }

  Json jdat = Json::object({
    {"kernels", kernels},
    {"objects", objects},
    {"programs", programs},
    {"binaries", jbinaries},
  });

  string str = jdat.dump();
  int jsz = str.length();

  FILE *f = fopen(filename, "wb");
  fwrite(&jsz, 1, sizeof(jsz), f);
  fwrite(str.data(), 1, jsz, f);
  for (auto &s : saved_buffers) {
    fwrite(s.data(), 1, s.length(), f);
  }
  fclose(f);
}

Json CLQueuedKernel::to_json() const {
  return Json::object {
    { "name", name },
    { "work_dim", (int)work_dim },
    { "global_work_size", Json::array { (int)global_work_size[0], (int)global_work_size[1], (int)global_work_size[2] } },
    { "local_work_size", Json::array { (int)local_work_size[0], (int)local_work_size[1], (int)local_work_size[2] } },
    { "num_args", (int)num_args },
    { "args", args },
    { "args_size", args_size },
  };
}

