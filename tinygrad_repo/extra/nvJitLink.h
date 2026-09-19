/*
 * NVIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * NVIDIA_COPYRIGHT_END
 */

#include <stdint.h>
#include <stdlib.h>

typedef enum {
  NVJITLINK_SUCCESS = 0,
  NVJITLINK_ERROR_UNRECOGNIZED_OPTION,
  NVJITLINK_ERROR_MISSING_ARCH,
  NVJITLINK_ERROR_INVALID_INPUT,
  NVJITLINK_ERROR_PTX_COMPILE,
  NVJITLINK_ERROR_NVVM_COMPILE,
  NVJITLINK_ERROR_INTERNAL
} nvJitLinkResult;

typedef enum {
  NVJITLINK_INPUT_NONE = 0,
  NVJITLINK_INPUT_CUBIN = 1,
  NVJITLINK_INPUT_PTX,
  NVJITLINK_INPUT_LTOIR,
  NVJITLINK_INPUT_FATBIN,
  NVJITLINK_INPUT_OBJECT,
  NVJITLINK_INPUT_LIBRARY
} nvJitLinkInputType;

typedef struct nvJitLink* nvJitLinkHandle;

nvJitLinkResult nvJitLinkCreate(nvJitLinkHandle *handle, uint32_t numOptions, const char **options);
nvJitLinkResult nvJitLinkDestroy(nvJitLinkHandle *handle);
nvJitLinkResult nvJitLinkAddData(nvJitLinkHandle handle, nvJitLinkInputType inputType, const void *data, size_t size, const char *name);
nvJitLinkResult nvJitLinkAddFile(nvJitLinkHandle handle, nvJitLinkInputType inputType, const char *fileName);
nvJitLinkResult nvJitLinkComplete(nvJitLinkHandle handle);
nvJitLinkResult nvJitLinkGetLinkedCubinSize(nvJitLinkHandle handle, size_t *size);
nvJitLinkResult nvJitLinkGetLinkedCubin(nvJitLinkHandle handle, void *cubin);
nvJitLinkResult nvJitLinkGetLinkedPtxSize(nvJitLinkHandle handle, size_t *size);
nvJitLinkResult nvJitLinkGetLinkedPtx(nvJitLinkHandle handle, char *ptx);
nvJitLinkResult nvJitLinkGetErrorLogSize(nvJitLinkHandle handle, size_t *size);
nvJitLinkResult nvJitLinkGetErrorLog(nvJitLinkHandle handle, char *log);
nvJitLinkResult nvJitLinkGetInfoLogSize(nvJitLinkHandle handle, size_t *size);
nvJitLinkResult nvJitLinkGetInfoLog(nvJitLinkHandle handle, char *log);
nvJitLinkResult nvJitLinkVersion(unsigned int *major, unsigned int *minor);
