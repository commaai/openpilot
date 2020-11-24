/*
 * Copyright (C) 2009 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef OMX_PLUGIN_BASE_H_

#define OMX_PLUGIN_BASE_H_

#include <sys/types.h>

#include <OMX_Component.h>

#include <utils/String8.h>
#include <utils/Vector.h>

namespace android {

struct OMXPluginBase {
    OMXPluginBase() {}
    virtual ~OMXPluginBase() {}

    virtual OMX_ERRORTYPE makeComponentInstance(
            const char *name,
            const OMX_CALLBACKTYPE *callbacks,
            OMX_PTR appData,
            OMX_COMPONENTTYPE **component) = 0;

    virtual OMX_ERRORTYPE destroyComponentInstance(
            OMX_COMPONENTTYPE *component) = 0;

    virtual OMX_ERRORTYPE enumerateComponents(
            OMX_STRING name,
            size_t size,
            OMX_U32 index) = 0;

    virtual OMX_ERRORTYPE getRolesOfComponent(
            const char *name,
            Vector<String8> *roles) = 0;

private:
    OMXPluginBase(const OMXPluginBase &);
    OMXPluginBase &operator=(const OMXPluginBase &);
};

}  // namespace android

#endif  // OMX_PLUGIN_BASE_H_
