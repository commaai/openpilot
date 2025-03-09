#pragma once

#include "common/clutil.h"
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <string>

cl_program cl_program_from_binary(cl_context ctx, cl_device_id device_id, const uint8_t* binary, size_t length, const char* args = nullptr);
const char* cl_get_error_string(int err);