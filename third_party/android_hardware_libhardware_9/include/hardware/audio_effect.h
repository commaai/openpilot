/*
 * Copyright (C) 2011 The Android Open Source Project
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


#ifndef ANDROID_AUDIO_EFFECT_H
#define ANDROID_AUDIO_EFFECT_H

#include <errno.h>
#include <stdint.h>
#include <strings.h>
#include <sys/cdefs.h>
#include <sys/types.h>

#include <cutils/bitops.h>

#include <system/audio_effect.h>


__BEGIN_DECLS


/////////////////////////////////////////////////
//      Common Definitions
/////////////////////////////////////////////////

#define EFFECT_MAKE_API_VERSION(M, m)  (((M)<<16) | ((m) & 0xFFFF))
#define EFFECT_API_VERSION_MAJOR(v)    ((v)>>16)
#define EFFECT_API_VERSION_MINOR(v)    ((m) & 0xFFFF)


/////////////////////////////////////////////////
//      Effect control interface
/////////////////////////////////////////////////

// Effect control interface version 2.0
#define EFFECT_CONTROL_API_VERSION EFFECT_MAKE_API_VERSION(2,0)

// Effect control interface structure: effect_interface_s
// The effect control interface is exposed by each effect engine implementation. It consists of
// a set of functions controlling the configuration, activation and process of the engine.
// The functions are grouped in a structure of type effect_interface_s.
//
// Effect control interface handle: effect_handle_t
// The effect_handle_t serves two purposes regarding the implementation of the effect engine:
// - 1 it is the address of a pointer to an effect_interface_s structure where the functions
// of the effect control API for a particular effect are located.
// - 2 it is the address of the context of a particular effect instance.
// A typical implementation in the effect library would define a structure as follows:
// struct effect_module_s {
//        const struct effect_interface_s *itfe;
//        effect_config_t config;
//        effect_context_t context;
// }
// The implementation of EffectCreate() function would then allocate a structure of this
// type and return its address as effect_handle_t
typedef struct effect_interface_s **effect_handle_t;

// Effect control interface definition
struct effect_interface_s {
    ////////////////////////////////////////////////////////////////////////////////
    //
    //    Function:       process
    //
    //    Description:    Effect process function. Takes input samples as specified
    //          (count and location) in input buffer descriptor and output processed
    //          samples as specified in output buffer descriptor. If the buffer descriptor
    //          is not specified the function must use either the buffer or the
    //          buffer provider function installed by the EFFECT_CMD_SET_CONFIG command.
    //          The effect framework will call the process() function after the EFFECT_CMD_ENABLE
    //          command is received and until the EFFECT_CMD_DISABLE is received. When the engine
    //          receives the EFFECT_CMD_DISABLE command it should turn off the effect gracefully
    //          and when done indicate that it is OK to stop calling the process() function by
    //          returning the -ENODATA status.
    //
    //    NOTE: the process() function implementation should be "real-time safe" that is
    //      it should not perform blocking calls: malloc/free, sleep, read/write/open/close,
    //      pthread_cond_wait/pthread_mutex_lock...
    //
    //    Input:
    //          self:       handle to the effect interface this function
    //              is called on.
    //          inBuffer:   buffer descriptor indicating where to read samples to process.
    //              If NULL, use the configuration passed by EFFECT_CMD_SET_CONFIG command.
    //
    //          outBuffer:   buffer descriptor indicating where to write processed samples.
    //              If NULL, use the configuration passed by EFFECT_CMD_SET_CONFIG command.
    //
    //    Output:
    //        returned value:    0 successful operation
    //                          -ENODATA the engine has finished the disable phase and the framework
    //                                  can stop calling process()
    //                          -EINVAL invalid interface handle or
    //                                  invalid input/output buffer description
    ////////////////////////////////////////////////////////////////////////////////
    int32_t (*process)(effect_handle_t self,
                       audio_buffer_t *inBuffer,
                       audio_buffer_t *outBuffer);
    ////////////////////////////////////////////////////////////////////////////////
    //
    //    Function:       command
    //
    //    Description:    Send a command and receive a response to/from effect engine.
    //
    //    Input:
    //          self:       handle to the effect interface this function
    //              is called on.
    //          cmdCode:    command code: the command can be a standardized command defined in
    //              effect_command_e (see below) or a proprietary command.
    //          cmdSize:    size of command in bytes
    //          pCmdData:   pointer to command data
    //          pReplyData: pointer to reply data
    //
    //    Input/Output:
    //          replySize: maximum size of reply data as input
    //                      actual size of reply data as output
    //
    //    Output:
    //          returned value: 0       successful operation
    //                          -EINVAL invalid interface handle or
    //                                  invalid command/reply size or format according to
    //                                  command code
    //              The return code should be restricted to indicate problems related to this API
    //              specification. Status related to the execution of a particular command should be
    //              indicated as part of the reply field.
    //
    //          *pReplyData updated with command response
    //
    ////////////////////////////////////////////////////////////////////////////////
    int32_t (*command)(effect_handle_t self,
                       uint32_t cmdCode,
                       uint32_t cmdSize,
                       void *pCmdData,
                       uint32_t *replySize,
                       void *pReplyData);
    ////////////////////////////////////////////////////////////////////////////////
    //
    //    Function:        get_descriptor
    //
    //    Description:    Returns the effect descriptor
    //
    //    Input:
    //          self:       handle to the effect interface this function
    //              is called on.
    //
    //    Input/Output:
    //          pDescriptor:    address where to return the effect descriptor.
    //
    //    Output:
    //        returned value:    0          successful operation.
    //                          -EINVAL     invalid interface handle or invalid pDescriptor
    //        *pDescriptor:     updated with the effect descriptor.
    //
    ////////////////////////////////////////////////////////////////////////////////
    int32_t (*get_descriptor)(effect_handle_t self,
                              effect_descriptor_t *pDescriptor);
    ////////////////////////////////////////////////////////////////////////////////
    //
    //    Function:       process_reverse
    //
    //    Description:    Process reverse stream function. This function is used to pass
    //          a reference stream to the effect engine. If the engine does not need a reference
    //          stream, this function pointer can be set to NULL.
    //          This function would typically implemented by an Echo Canceler.
    //
    //    Input:
    //          self:       handle to the effect interface this function
    //              is called on.
    //          inBuffer:   buffer descriptor indicating where to read samples to process.
    //              If NULL, use the configuration passed by EFFECT_CMD_SET_CONFIG_REVERSE command.
    //
    //          outBuffer:   buffer descriptor indicating where to write processed samples.
    //              If NULL, use the configuration passed by EFFECT_CMD_SET_CONFIG_REVERSE command.
    //              If the buffer and buffer provider in the configuration received by
    //              EFFECT_CMD_SET_CONFIG_REVERSE are also NULL, do not return modified reverse
    //              stream data
    //
    //    Output:
    //        returned value:    0 successful operation
    //                          -ENODATA the engine has finished the disable phase and the framework
    //                                  can stop calling process_reverse()
    //                          -EINVAL invalid interface handle or
    //                                  invalid input/output buffer description
    ////////////////////////////////////////////////////////////////////////////////
    int32_t (*process_reverse)(effect_handle_t self,
                               audio_buffer_t *inBuffer,
                               audio_buffer_t *outBuffer);
};

/////////////////////////////////////////////////
//      Effect library interface
/////////////////////////////////////////////////

// Effect library interface version 3.0
// Note that EffectsFactory.c only checks the major version component, so changes to the minor
// number can only be used for fully backwards compatible changes
#define EFFECT_LIBRARY_API_VERSION EFFECT_MAKE_API_VERSION(3,0)

#define AUDIO_EFFECT_LIBRARY_TAG ((('A') << 24) | (('E') << 16) | (('L') << 8) | ('T'))

// Every effect library must have a data structure named AUDIO_EFFECT_LIBRARY_INFO_SYM
// and the fields of this data structure must begin with audio_effect_library_t

typedef struct audio_effect_library_s {
    // tag must be initialized to AUDIO_EFFECT_LIBRARY_TAG
    uint32_t tag;
    // Version of the effect library API : 0xMMMMmmmm MMMM: Major, mmmm: minor
    uint32_t version;
    // Name of this library
    const char *name;
    // Author/owner/implementor of the library
    const char *implementor;

    ////////////////////////////////////////////////////////////////////////////////
    //
    //    Function:        create_effect
    //
    //    Description:    Creates an effect engine of the specified implementation uuid and
    //          returns an effect control interface on this engine. The function will allocate the
    //          resources for an instance of the requested effect engine and return
    //          a handle on the effect control interface.
    //
    //    Input:
    //          uuid:    pointer to the effect uuid.
    //          sessionId:  audio session to which this effect instance will be attached.
    //              All effects created with the same session ID are connected in series and process
    //              the same signal stream. Knowing that two effects are part of the same effect
    //              chain can help the library implement some kind of optimizations.
    //          ioId:   identifies the output or input stream this effect is directed to in
    //              audio HAL.
    //              For future use especially with tunneled HW accelerated effects
    //
    //    Input/Output:
    //          pHandle:        address where to return the effect interface handle.
    //
    //    Output:
    //        returned value:    0          successful operation.
    //                          -ENODEV     library failed to initialize
    //                          -EINVAL     invalid pEffectUuid or pHandle
    //                          -ENOENT     no effect with this uuid found
    //        *pHandle:         updated with the effect interface handle.
    //
    ////////////////////////////////////////////////////////////////////////////////
    int32_t (*create_effect)(const effect_uuid_t *uuid,
                             int32_t sessionId,
                             int32_t ioId,
                             effect_handle_t *pHandle);

    ////////////////////////////////////////////////////////////////////////////////
    //
    //    Function:        release_effect
    //
    //    Description:    Releases the effect engine whose handle is given as argument.
    //          All resources allocated to this particular instance of the effect are
    //          released.
    //
    //    Input:
    //          handle:         handle on the effect interface to be released.
    //
    //    Output:
    //        returned value:    0          successful operation.
    //                          -ENODEV     library failed to initialize
    //                          -EINVAL     invalid interface handle
    //
    ////////////////////////////////////////////////////////////////////////////////
    int32_t (*release_effect)(effect_handle_t handle);

    ////////////////////////////////////////////////////////////////////////////////
    //
    //    Function:        get_descriptor
    //
    //    Description:    Returns the descriptor of the effect engine which implementation UUID is
    //          given as argument.
    //
    //    Input/Output:
    //          uuid:           pointer to the effect uuid.
    //          pDescriptor:    address where to return the effect descriptor.
    //
    //    Output:
    //        returned value:    0          successful operation.
    //                          -ENODEV     library failed to initialize
    //                          -EINVAL     invalid pDescriptor or uuid
    //        *pDescriptor:     updated with the effect descriptor.
    //
    ////////////////////////////////////////////////////////////////////////////////
    int32_t (*get_descriptor)(const effect_uuid_t *uuid,
                              effect_descriptor_t *pDescriptor);
} audio_effect_library_t;

// Name of the hal_module_info
#define AUDIO_EFFECT_LIBRARY_INFO_SYM         AELI

// Name of the hal_module_info as a string
#define AUDIO_EFFECT_LIBRARY_INFO_SYM_AS_STR  "AELI"

__END_DECLS

#endif  // ANDROID_AUDIO_EFFECT_H
