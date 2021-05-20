/*
 * Copyright (c) 2009 The Khronos Group Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 */

/** OMX_CoreExt.h - OpenMax IL version 1.1.2
 * The OMX_CoreExt header file contains extensions to the definitions used
 * by both the application and the component to access common items.
 */

#ifndef OMX_CoreExt_h
#define OMX_CoreExt_h

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/* Each OMX header shall include all required header files to allow the
 * header to compile without errors.  The includes below are required
 * for this header file to compile successfully
 */
#include <OMX_Core.h>


/** Event type extensions. */
typedef enum OMX_EVENTEXTTYPE
{
    OMX_EventIndexSettingChanged = OMX_EventKhronosExtensions, /**< component signals the IL client of a change
                                                                    in a param, config, or extension */
    OMX_EventExtMax = 0x7FFFFFFF
} OMX_EVENTEXTTYPE;


/** Enable or disable a callback event. */
typedef struct OMX_CONFIG_CALLBACKREQUESTTYPE {
    OMX_U32 nSize;              /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion;   /**< OMX specification version information */
    OMX_U32 nPortIndex;         /**< port that this structure applies to */
    OMX_INDEXTYPE nIndex;       /**< the index the callback is requested for */
    OMX_BOOL bEnable;           /**< enable (OMX_TRUE) or disable (OMX_FALSE) the callback */
} OMX_CONFIG_CALLBACKREQUESTTYPE;

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* OMX_CoreExt_h */
/* File EOF */
