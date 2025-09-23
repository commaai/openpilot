/**********************************************************************************
 * Copyright (c) 2008-2015 The Khronos Group Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and/or associated documentation files (the
 * "Materials"), to deal in the Materials without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Materials, and to
 * permit persons to whom the Materials are furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Materials.
 *
 * MODIFICATIONS TO THIS FILE MAY MEAN IT NO LONGER ACCURATELY REFLECTS
 * KHRONOS STANDARDS. THE UNMODIFIED, NORMATIVE VERSIONS OF KHRONOS
 * SPECIFICATIONS AND HEADER INFORMATION ARE LOCATED AT
 *    https://www.khronos.org/registry/
 *
 * THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
 **********************************************************************************/

#ifndef __OPENCL_CL_GL_H
#define __OPENCL_CL_GL_H

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif	

#ifdef __cplusplus
extern "C" {
#endif

typedef cl_uint     cl_gl_object_type;
typedef cl_uint     cl_gl_texture_info;
typedef cl_uint     cl_gl_platform_info;
typedef struct __GLsync *cl_GLsync;

/* cl_gl_object_type = 0x2000 - 0x200F enum values are currently taken           */
#define CL_GL_OBJECT_BUFFER                     0x2000
#define CL_GL_OBJECT_TEXTURE2D                  0x2001
#define CL_GL_OBJECT_TEXTURE3D                  0x2002
#define CL_GL_OBJECT_RENDERBUFFER               0x2003
#define CL_GL_OBJECT_TEXTURE2D_ARRAY            0x200E
#define CL_GL_OBJECT_TEXTURE1D                  0x200F
#define CL_GL_OBJECT_TEXTURE1D_ARRAY            0x2010
#define CL_GL_OBJECT_TEXTURE_BUFFER             0x2011

/* cl_gl_texture_info           */
#define CL_GL_TEXTURE_TARGET                    0x2004
#define CL_GL_MIPMAP_LEVEL                      0x2005
#define CL_GL_NUM_SAMPLES                       0x2012


extern CL_API_ENTRY cl_mem CL_API_CALL
clCreateFromGLBuffer(cl_context     /* context */,
                     cl_mem_flags   /* flags */,
                     cl_GLuint      /* bufobj */,
                     int *          /* errcode_ret */) CL_API_SUFFIX__VERSION_1_0;

extern CL_API_ENTRY cl_mem CL_API_CALL
clCreateFromGLTexture(cl_context      /* context */,
                      cl_mem_flags    /* flags */,
                      cl_GLenum       /* target */,
                      cl_GLint        /* miplevel */,
                      cl_GLuint       /* texture */,
                      cl_int *        /* errcode_ret */) CL_API_SUFFIX__VERSION_1_2;
    
extern CL_API_ENTRY cl_mem CL_API_CALL
clCreateFromGLRenderbuffer(cl_context   /* context */,
                           cl_mem_flags /* flags */,
                           cl_GLuint    /* renderbuffer */,
                           cl_int *     /* errcode_ret */) CL_API_SUFFIX__VERSION_1_0;

extern CL_API_ENTRY cl_int CL_API_CALL
clGetGLObjectInfo(cl_mem                /* memobj */,
                  cl_gl_object_type *   /* gl_object_type */,
                  cl_GLuint *           /* gl_object_name */) CL_API_SUFFIX__VERSION_1_0;
                  
extern CL_API_ENTRY cl_int CL_API_CALL
clGetGLTextureInfo(cl_mem               /* memobj */,
                   cl_gl_texture_info   /* param_name */,
                   size_t               /* param_value_size */,
                   void *               /* param_value */,
                   size_t *             /* param_value_size_ret */) CL_API_SUFFIX__VERSION_1_0;

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueAcquireGLObjects(cl_command_queue      /* command_queue */,
                          cl_uint               /* num_objects */,
                          const cl_mem *        /* mem_objects */,
                          cl_uint               /* num_events_in_wait_list */,
                          const cl_event *      /* event_wait_list */,
                          cl_event *            /* event */) CL_API_SUFFIX__VERSION_1_0;

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueReleaseGLObjects(cl_command_queue      /* command_queue */,
                          cl_uint               /* num_objects */,
                          const cl_mem *        /* mem_objects */,
                          cl_uint               /* num_events_in_wait_list */,
                          const cl_event *      /* event_wait_list */,
                          cl_event *            /* event */) CL_API_SUFFIX__VERSION_1_0;


/* Deprecated OpenCL 1.1 APIs */
extern CL_API_ENTRY CL_EXT_PREFIX__VERSION_1_1_DEPRECATED cl_mem CL_API_CALL
clCreateFromGLTexture2D(cl_context      /* context */,
                        cl_mem_flags    /* flags */,
                        cl_GLenum       /* target */,
                        cl_GLint        /* miplevel */,
                        cl_GLuint       /* texture */,
                        cl_int *        /* errcode_ret */) CL_EXT_SUFFIX__VERSION_1_1_DEPRECATED;
    
extern CL_API_ENTRY CL_EXT_PREFIX__VERSION_1_1_DEPRECATED cl_mem CL_API_CALL
clCreateFromGLTexture3D(cl_context      /* context */,
                        cl_mem_flags    /* flags */,
                        cl_GLenum       /* target */,
                        cl_GLint        /* miplevel */,
                        cl_GLuint       /* texture */,
                        cl_int *        /* errcode_ret */) CL_EXT_SUFFIX__VERSION_1_1_DEPRECATED;
    
/* cl_khr_gl_sharing extension  */
    
#define cl_khr_gl_sharing 1
    
typedef cl_uint     cl_gl_context_info;
    
/* Additional Error Codes  */
#define CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR  -1000
    
/* cl_gl_context_info  */
#define CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR    0x2006
#define CL_DEVICES_FOR_GL_CONTEXT_KHR           0x2007
    
/* Additional cl_context_properties  */
#define CL_GL_CONTEXT_KHR                       0x2008
#define CL_EGL_DISPLAY_KHR                      0x2009
#define CL_GLX_DISPLAY_KHR                      0x200A
#define CL_WGL_HDC_KHR                          0x200B
#define CL_CGL_SHAREGROUP_KHR                   0x200C
    
extern CL_API_ENTRY cl_int CL_API_CALL
clGetGLContextInfoKHR(const cl_context_properties * /* properties */,
                      cl_gl_context_info            /* param_name */,
                      size_t                        /* param_value_size */,
                      void *                        /* param_value */,
                      size_t *                      /* param_value_size_ret */) CL_API_SUFFIX__VERSION_1_0;
    
typedef CL_API_ENTRY cl_int (CL_API_CALL *clGetGLContextInfoKHR_fn)(
    const cl_context_properties * properties,
    cl_gl_context_info            param_name,
    size_t                        param_value_size,
    void *                        param_value,
    size_t *                      param_value_size_ret);

#ifdef __cplusplus
}
#endif

#endif  /* __OPENCL_CL_GL_H */
