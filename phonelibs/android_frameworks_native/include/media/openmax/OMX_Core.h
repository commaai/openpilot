/* ------------------------------------------------------------------
 * Copyright (C) 1998-2009 PacketVideo
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 * express or implied.
 * See the License for the specific language governing permissions
 * and limitations under the License.
 * -------------------------------------------------------------------
 */
/*
 * Copyright (c) 2008 The Khronos Group Inc.
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

/** OMX_Core.h - OpenMax IL version 1.1.2
 *  The OMX_Core header file contains the definitions used by both the
 *  application and the component to access common items.
 */

#ifndef OMX_Core_h
#define OMX_Core_h

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */


/* Each OMX header shall include all required header files to allow the
 *  header to compile without errors.  The includes below are required
 *  for this header file to compile successfully
 */

#include <OMX_Index.h>


/** The OMX_COMMANDTYPE enumeration is used to specify the action in the
 *  OMX_SendCommand macro.
 *  @ingroup core
 */
typedef enum OMX_COMMANDTYPE
{
    OMX_CommandStateSet,    /**< Change the component state */
    OMX_CommandFlush,       /**< Flush the data queue(s) of a component */
    OMX_CommandPortDisable, /**< Disable a port on a component. */
    OMX_CommandPortEnable,  /**< Enable a port on a component. */
    OMX_CommandMarkBuffer,  /**< Mark a component/buffer for observation */
    OMX_CommandKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
    OMX_CommandVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_CommandMax = 0X7FFFFFFF
} OMX_COMMANDTYPE;



/** The OMX_STATETYPE enumeration is used to indicate or change the component
 *  state.  This enumeration reflects the current state of the component when
 *  used with the OMX_GetState macro or becomes the parameter in a state change
 *  command when used with the OMX_SendCommand macro.
 *
 *  The component will be in the Loaded state after the component is initially
 *  loaded into memory.  In the Loaded state, the component is not allowed to
 *  allocate or hold resources other than to build it's internal parameter
 *  and configuration tables.  The application will send one or more
 *  SetParameters/GetParameters and SetConfig/GetConfig commands to the
 *  component and the component will record each of these parameter and
 *  configuration changes for use later.  When the application sends the
 *  Idle command, the component will acquire the resources needed for the
 *  specified configuration and will transition to the idle state if the
 *  allocation is successful.  If the component cannot successfully
 *  transition to the idle state for any reason, the state of the component
 *  shall be fully rolled back to the Loaded state (e.g. all allocated
 *  resources shall be released).  When the component receives the command
 *  to go to the Executing state, it shall begin processing buffers by
 *  sending all input buffers it holds to the application.  While
 *  the component is in the Idle state, the application may also send the
 *  Pause command.  If the component receives the pause command while in the
 *  Idle state, the component shall send all input buffers it holds to the
 *  application, but shall not begin processing buffers.  This will allow the
 *  application to prefill buffers.
 *
 *  @ingroup comp
 */

typedef enum OMX_STATETYPE
{
    OMX_StateInvalid,      /**< component has detected that it's internal data
                                structures are corrupted to the point that
                                it cannot determine it's state properly */
    OMX_StateLoaded,      /**< component has been loaded but has not completed
                                initialization.  The OMX_SetParameter macro
                                and the OMX_GetParameter macro are the only
                                valid macros allowed to be sent to the
                                component in this state. */
    OMX_StateIdle,        /**< component initialization has been completed
                                successfully and the component is ready to
                                to start. */
    OMX_StateExecuting,   /**< component has accepted the start command and
                                is processing data (if data is available) */
    OMX_StatePause,       /**< component has received pause command */
    OMX_StateWaitForResources, /**< component is waiting for resources, either after
                                preemption or before it gets the resources requested.
                                See specification for complete details. */
    OMX_StateKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
    OMX_StateVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_StateMax = 0X7FFFFFFF
} OMX_STATETYPE;

/** The OMX_ERRORTYPE enumeration defines the standard OMX Errors.  These
 *  errors should cover most of the common failure cases.  However,
 *  vendors are free to add additional error messages of their own as
 *  long as they follow these rules:
 *  1.  Vendor error messages shall be in the range of 0x90000000 to
 *      0x9000FFFF.
 *  2.  Vendor error messages shall be defined in a header file provided
 *      with the component.  No error messages are allowed that are
 *      not defined.
 */
typedef enum OMX_ERRORTYPE
{
  OMX_ErrorNone = 0,

  /** There were insufficient resources to perform the requested operation */
  OMX_ErrorInsufficientResources = (OMX_S32) 0x80001000,

  /** There was an error, but the cause of the error could not be determined */
  OMX_ErrorUndefined = (OMX_S32) 0x80001001,

  /** The component name string was not valid */
  OMX_ErrorInvalidComponentName = (OMX_S32) 0x80001002,

  /** No component with the specified name string was found */
  OMX_ErrorComponentNotFound = (OMX_S32) 0x80001003,

  /** The component specified did not have a "OMX_ComponentInit" or
      "OMX_ComponentDeInit entry point */
  OMX_ErrorInvalidComponent = (OMX_S32) 0x80001004,

  /** One or more parameters were not valid */
  OMX_ErrorBadParameter = (OMX_S32) 0x80001005,

  /** The requested function is not implemented */
  OMX_ErrorNotImplemented = (OMX_S32) 0x80001006,

  /** The buffer was emptied before the next buffer was ready */
  OMX_ErrorUnderflow = (OMX_S32) 0x80001007,

  /** The buffer was not available when it was needed */
  OMX_ErrorOverflow = (OMX_S32) 0x80001008,

  /** The hardware failed to respond as expected */
  OMX_ErrorHardware = (OMX_S32) 0x80001009,

  /** The component is in the state OMX_StateInvalid */
  OMX_ErrorInvalidState = (OMX_S32) 0x8000100A,

  /** Stream is found to be corrupt */
  OMX_ErrorStreamCorrupt = (OMX_S32) 0x8000100B,

  /** Ports being connected are not compatible */
  OMX_ErrorPortsNotCompatible = (OMX_S32) 0x8000100C,

  /** Resources allocated to an idle component have been
      lost resulting in the component returning to the loaded state */
  OMX_ErrorResourcesLost = (OMX_S32) 0x8000100D,

  /** No more indicies can be enumerated */
  OMX_ErrorNoMore = (OMX_S32) 0x8000100E,

  /** The component detected a version mismatch */
  OMX_ErrorVersionMismatch = (OMX_S32) 0x8000100F,

  /** The component is not ready to return data at this time */
  OMX_ErrorNotReady = (OMX_S32) 0x80001010,

  /** There was a timeout that occurred */
  OMX_ErrorTimeout = (OMX_S32) 0x80001011,

  /** This error occurs when trying to transition into the state you are already in */
  OMX_ErrorSameState = (OMX_S32) 0x80001012,

  /** Resources allocated to an executing or paused component have been
      preempted, causing the component to return to the idle state */
  OMX_ErrorResourcesPreempted = (OMX_S32) 0x80001013,

  /** A non-supplier port sends this error to the IL client (via the EventHandler callback)
      during the allocation of buffers (on a transition from the LOADED to the IDLE state or
      on a port restart) when it deems that it has waited an unusually long time for the supplier
      to send it an allocated buffer via a UseBuffer call. */
  OMX_ErrorPortUnresponsiveDuringAllocation = (OMX_S32) 0x80001014,

  /** A non-supplier port sends this error to the IL client (via the EventHandler callback)
      during the deallocation of buffers (on a transition from the IDLE to LOADED state or
      on a port stop) when it deems that it has waited an unusually long time for the supplier
      to request the deallocation of a buffer header via a FreeBuffer call. */
  OMX_ErrorPortUnresponsiveDuringDeallocation = (OMX_S32) 0x80001015,

  /** A supplier port sends this error to the IL client (via the EventHandler callback)
      during the stopping of a port (either on a transition from the IDLE to LOADED
      state or a port stop) when it deems that it has waited an unusually long time for
      the non-supplier to return a buffer via an EmptyThisBuffer or FillThisBuffer call. */
  OMX_ErrorPortUnresponsiveDuringStop = (OMX_S32) 0x80001016,

  /** Attempting a state transtion that is not allowed */
  OMX_ErrorIncorrectStateTransition = (OMX_S32) 0x80001017,

  /* Attempting a command that is not allowed during the present state. */
  OMX_ErrorIncorrectStateOperation = (OMX_S32) 0x80001018,

  /** The values encapsulated in the parameter or config structure are not supported. */
  OMX_ErrorUnsupportedSetting = (OMX_S32) 0x80001019,

  /** The parameter or config indicated by the given index is not supported. */
  OMX_ErrorUnsupportedIndex = (OMX_S32) 0x8000101A,

  /** The port index supplied is incorrect. */
  OMX_ErrorBadPortIndex = (OMX_S32) 0x8000101B,

  /** The port has lost one or more of its buffers and it thus unpopulated. */
  OMX_ErrorPortUnpopulated = (OMX_S32) 0x8000101C,

  /** Component suspended due to temporary loss of resources */
  OMX_ErrorComponentSuspended = (OMX_S32) 0x8000101D,

  /** Component suspended due to an inability to acquire dynamic resources */
  OMX_ErrorDynamicResourcesUnavailable = (OMX_S32) 0x8000101E,

  /** When the macroblock error reporting is enabled the component returns new error
  for every frame that has errors */
  OMX_ErrorMbErrorsInFrame = (OMX_S32) 0x8000101F,

  /** A component reports this error when it cannot parse or determine the format of an input stream. */
  OMX_ErrorFormatNotDetected = (OMX_S32) 0x80001020,

  /** The content open operation failed. */
  OMX_ErrorContentPipeOpenFailed = (OMX_S32) 0x80001021,

  /** The content creation operation failed. */
  OMX_ErrorContentPipeCreationFailed = (OMX_S32) 0x80001022,

  /** Separate table information is being used */
  OMX_ErrorSeperateTablesUsed = (OMX_S32) 0x80001023,

  /** Tunneling is unsupported by the component*/
  OMX_ErrorTunnelingUnsupported = (OMX_S32) 0x80001024,

  OMX_ErrorKhronosExtensions = (OMX_S32)0x8F000000, /**< Reserved region for introducing Khronos Standard Extensions */
  OMX_ErrorVendorStartUnused = (OMX_S32)0x90000000, /**< Reserved region for introducing Vendor Extensions */
  OMX_ErrorMax = 0x7FFFFFFF
} OMX_ERRORTYPE;

/** @ingroup core */
typedef OMX_ERRORTYPE (* OMX_COMPONENTINITTYPE)(OMX_IN  OMX_HANDLETYPE hComponent);

/** @ingroup core */
typedef struct OMX_COMPONENTREGISTERTYPE
{
  const char          * pName;       /* Component name, 128 byte limit (including '\0') applies */
  OMX_COMPONENTINITTYPE pInitialize; /* Component instance initialization function */
} OMX_COMPONENTREGISTERTYPE;

/** @ingroup core */
extern OMX_COMPONENTREGISTERTYPE OMX_ComponentRegistered[];

/** @ingroup rpm */
typedef struct OMX_PRIORITYMGMTTYPE {
 OMX_U32 nSize;             /**< size of the structure in bytes */
 OMX_VERSIONTYPE nVersion;  /**< OMX specification version information */
 OMX_U32 nGroupPriority;            /**< Priority of the component group */
 OMX_U32 nGroupID;                  /**< ID of the component group */
} OMX_PRIORITYMGMTTYPE;

/* Component name and Role names are limited to 128 characters including the terminating '\0'. */
#define OMX_MAX_STRINGNAME_SIZE 128

/** @ingroup comp */
typedef struct OMX_PARAM_COMPONENTROLETYPE {
    OMX_U32 nSize;              /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion;   /**< OMX specification version information */
    OMX_U8 cRole[OMX_MAX_STRINGNAME_SIZE];  /**< name of standard component which defines component role */
} OMX_PARAM_COMPONENTROLETYPE;

/** End of Stream Buffer Flag:
  *
  * A component sets EOS when it has no more data to emit on a particular
  * output port. Thus an output port shall set EOS on the last buffer it
  * emits. A component's determination of when an output port should
  * cease sending data is implemenation specific.
  * @ingroup buf
  */

#define OMX_BUFFERFLAG_EOS 0x00000001

/** Start Time Buffer Flag:
 *
 * The source of a stream (e.g. a demux component) sets the STARTTIME
 * flag on the buffer that contains the starting timestamp for the
 * stream. The starting timestamp corresponds to the first data that
 * should be displayed at startup or after a seek.
 * The first timestamp of the stream is not necessarily the start time.
 * For instance, in the case of a seek to a particular video frame,
 * the target frame may be an interframe. Thus the first buffer of
 * the stream will be the intra-frame preceding the target frame and
 * the starttime will occur with the target frame (with any other
 * required frames required to reconstruct the target intervening).
 *
 * The STARTTIME flag is directly associated with the buffer's
 * timestamp ' thus its association to buffer data and its
 * propagation is identical to the timestamp's.
 *
 * When a Sync Component client receives a buffer with the
 * STARTTIME flag it shall perform a SetConfig on its sync port
 * using OMX_ConfigTimeClientStartTime and passing the buffer's
 * timestamp.
 *
 * @ingroup buf
 */

#define OMX_BUFFERFLAG_STARTTIME 0x00000002



/** Decode Only Buffer Flag:
 *
 * The source of a stream (e.g. a demux component) sets the DECODEONLY
 * flag on any buffer that should shall be decoded but should not be
 * displayed. This flag is used, for instance, when a source seeks to
 * a target interframe that requires the decode of frames preceding the
 * target to facilitate the target's reconstruction. In this case the
 * source would emit the frames preceding the target downstream
 * but mark them as decode only.
 *
 * The DECODEONLY is associated with buffer data and propagated in a
 * manner identical to the buffer timestamp.
 *
 * A component that renders data should ignore all buffers with
 * the DECODEONLY flag set.
 *
 * @ingroup buf
 */

#define OMX_BUFFERFLAG_DECODEONLY 0x00000004


/* Data Corrupt Flag: This flag is set when the IL client believes the data in the associated buffer is corrupt
 * @ingroup buf
 */

#define OMX_BUFFERFLAG_DATACORRUPT 0x00000008

/* End of Frame: The buffer contains exactly one end of frame and no data
 *  occurs after the end of frame. This flag is an optional hint. The absence
 *  of this flag does not imply the absence of an end of frame within the buffer.
 * @ingroup buf
*/
#define OMX_BUFFERFLAG_ENDOFFRAME 0x00000010

/* Sync Frame Flag: This flag is set when the buffer content contains a coded sync frame '
 *  a frame that has no dependency on any other frame information
 *  @ingroup buf
 */
#define OMX_BUFFERFLAG_SYNCFRAME 0x00000020

/* Extra data present flag: there is extra data appended to the data stream
 * residing in the buffer
 * @ingroup buf
 */
#define OMX_BUFFERFLAG_EXTRADATA 0x00000040

/** Codec Config Buffer Flag:
* OMX_BUFFERFLAG_CODECCONFIG is an optional flag that is set by an
* output port when all bytes in the buffer form part or all of a set of
* codec specific configuration data.  Examples include SPS/PPS nal units
* for OMX_VIDEO_CodingAVC or AudioSpecificConfig data for
* OMX_AUDIO_CodingAAC.  Any component that for a given stream sets
* OMX_BUFFERFLAG_CODECCONFIG shall not mix codec configuration bytes
* with frame data in the same buffer, and shall send all buffers
* containing codec configuration bytes before any buffers containing
* frame data that those configurations bytes describe.
* If the stream format for a particular codec has a frame specific
* header at the start of each frame, for example OMX_AUDIO_CodingMP3 or
* OMX_AUDIO_CodingAAC in ADTS mode, then these shall be presented as
* normal without setting OMX_BUFFERFLAG_CODECCONFIG.
 * @ingroup buf
 */
#define OMX_BUFFERFLAG_CODECCONFIG 0x00000080



/** @ingroup buf */
typedef struct OMX_BUFFERHEADERTYPE
{
    OMX_U32 nSize;              /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion;   /**< OMX specification version information */
    OMX_U8* pBuffer;            /**< Pointer to actual block of memory
                                     that is acting as the buffer */
    OMX_U32 nAllocLen;          /**< size of the buffer allocated, in bytes */
    OMX_U32 nFilledLen;         /**< number of bytes currently in the
                                     buffer */
    OMX_U32 nOffset;            /**< start offset of valid data in bytes from
                                     the start of the buffer */
    OMX_PTR pAppPrivate;        /**< pointer to any data the application
                                     wants to associate with this buffer */
    OMX_PTR pPlatformPrivate;   /**< pointer to any data the platform
                                     wants to associate with this buffer */
    OMX_PTR pInputPortPrivate;  /**< pointer to any data the input port
                                     wants to associate with this buffer */
    OMX_PTR pOutputPortPrivate; /**< pointer to any data the output port
                                     wants to associate with this buffer */
    OMX_HANDLETYPE hMarkTargetComponent; /**< The component that will generate a
                                              mark event upon processing this buffer. */
    OMX_PTR pMarkData;          /**< Application specific data associated with
                                     the mark sent on a mark event to disambiguate
                                     this mark from others. */
    OMX_U32 nTickCount;         /**< Optional entry that the component and
                                     application can update with a tick count
                                     when they access the component.  This
                                     value should be in microseconds.  Since
                                     this is a value relative to an arbitrary
                                     starting point, this value cannot be used
                                     to determine absolute time.  This is an
                                     optional entry and not all components
                                     will update it.*/
 OMX_TICKS nTimeStamp;          /**< Timestamp corresponding to the sample
                                     starting at the first logical sample
                                     boundary in the buffer. Timestamps of
                                     successive samples within the buffer may
                                     be inferred by adding the duration of the
                                     of the preceding buffer to the timestamp
                                     of the preceding buffer.*/
  OMX_U32     nFlags;           /**< buffer specific flags */
  OMX_U32 nOutputPortIndex;     /**< The index of the output port (if any) using
                                     this buffer */
  OMX_U32 nInputPortIndex;      /**< The index of the input port (if any) using
                                     this buffer */
} OMX_BUFFERHEADERTYPE;

/** The OMX_EXTRADATATYPE enumeration is used to define the
 * possible extra data payload types.
 * NB: this enum is binary backwards compatible with the previous
 * OMX_EXTRADATA_QUANT define.  This should be replaced with
 * OMX_ExtraDataQuantization.
 */
typedef enum OMX_EXTRADATATYPE
{
   OMX_ExtraDataNone = 0,                       /**< Indicates that no more extra data sections follow */
   OMX_ExtraDataQuantization,                   /**< The data payload contains quantization data */
   OMX_ExtraDataKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
   OMX_ExtraDataVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
   OMX_ExtraDataMax = 0x7FFFFFFF
} OMX_EXTRADATATYPE;


typedef struct OMX_OTHER_EXTRADATATYPE  {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_U32 nPortIndex;
    OMX_EXTRADATATYPE eType;       /* Extra Data type */
    OMX_U32 nDataSize;   /* Size of the supporting data to follow */
    OMX_U8  data[1];     /* Supporting data hint  */
} OMX_OTHER_EXTRADATATYPE;

/** @ingroup comp */
typedef struct OMX_PORT_PARAM_TYPE {
    OMX_U32 nSize;              /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion;   /**< OMX specification version information */
    OMX_U32 nPorts;             /**< The number of ports for this component */
    OMX_U32 nStartPortNumber;   /** first port number for this type of port */
} OMX_PORT_PARAM_TYPE;

/** @ingroup comp */
typedef enum OMX_EVENTTYPE
{
    OMX_EventCmdComplete,         /**< component has sucessfully completed a command */
    OMX_EventError,               /**< component has detected an error condition */
    OMX_EventMark,                /**< component has detected a buffer mark */
    OMX_EventPortSettingsChanged, /**< component is reported a port settings change */
    OMX_EventBufferFlag,          /**< component has detected an EOS */
    OMX_EventResourcesAcquired,   /**< component has been granted resources and is
                                       automatically starting the state change from
                                       OMX_StateWaitForResources to OMX_StateIdle. */
    OMX_EventComponentResumed,     /**< Component resumed due to reacquisition of resources */
    OMX_EventDynamicResourcesAvailable, /**< Component has acquired previously unavailable dynamic resources */
    OMX_EventPortFormatDetected,      /**< Component has detected a supported format. */
    OMX_EventKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
    OMX_EventVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */

    /** Event when tunneled decoder has rendered an output or reached EOS
     *  nData1 must contain the number of timestamps returned
     *  pEventData must point to an array of the OMX_VIDEO_RENDEREVENTTYPE structs containing the
     *  render-timestamps of each frame. Component may batch rendered timestamps using this event,
     *  but must signal the event no more than 40ms after the first frame in the batch. The frames
     *  must be ordered by system timestamp inside and across batches.
     *
     *  If component is doing frame-rate conversion, it must signal the render time of each
     *  converted frame, and must interpolate media timestamps for in-between frames.
     *
     *  When the component reached EOS, it must signal an EOS timestamp using the same mechanism.
     *  This is in addition to the timestamp of the last rendered frame, and should follow that
     *  frame.
     */
    OMX_EventOutputRendered = 0x7F000001,
    OMX_EventMax = 0x7FFFFFFF
} OMX_EVENTTYPE;

typedef struct OMX_CALLBACKTYPE
{
    /** The EventHandler method is used to notify the application when an
        event of interest occurs.  Events are defined in the OMX_EVENTTYPE
        enumeration.  Please see that enumeration for details of what will
        be returned for each type of event. Callbacks should not return
        an error to the component, so if an error occurs, the application
        shall handle it internally.  This is a blocking call.

        The application should return from this call within 5 msec to avoid
        blocking the component for an excessively long period of time.

        @param hComponent
            handle of the component to access.  This is the component
            handle returned by the call to the GetHandle function.
        @param pAppData
            pointer to an application defined value that was provided in the
            pAppData parameter to the OMX_GetHandle method for the component.
            This application defined value is provided so that the application
            can have a component specific context when receiving the callback.
        @param eEvent
            Event that the component wants to notify the application about.
        @param nData1
            nData will be the OMX_ERRORTYPE for an error event and will be
            an OMX_COMMANDTYPE for a command complete event and OMX_INDEXTYPE for a OMX_PortSettingsChanged event.
         @param nData2
            nData2 will hold further information related to the event. Can be OMX_STATETYPE for
            a OMX_CommandStateSet command or port index for a OMX_PortSettingsChanged event.
            Default value is 0 if not used. )
        @param pEventData
            Pointer to additional event-specific data (see spec for meaning).
      */

   OMX_ERRORTYPE (*EventHandler)(
        OMX_IN OMX_HANDLETYPE hComponent,
        OMX_IN OMX_PTR pAppData,
        OMX_IN OMX_EVENTTYPE eEvent,
        OMX_IN OMX_U32 nData1,
        OMX_IN OMX_U32 nData2,
        OMX_IN OMX_PTR pEventData);

    /** The EmptyBufferDone method is used to return emptied buffers from an
        input port back to the application for reuse.  This is a blocking call
        so the application should not attempt to refill the buffers during this
        call, but should queue them and refill them in another thread.  There
        is no error return, so the application shall handle any errors generated
        internally.

        The application should return from this call within 5 msec.

        @param hComponent
            handle of the component to access.  This is the component
            handle returned by the call to the GetHandle function.
        @param pAppData
            pointer to an application defined value that was provided in the
            pAppData parameter to the OMX_GetHandle method for the component.
            This application defined value is provided so that the application
            can have a component specific context when receiving the callback.
        @param pBuffer
            pointer to an OMX_BUFFERHEADERTYPE structure allocated with UseBuffer
            or AllocateBuffer indicating the buffer that was emptied.
        @ingroup buf
     */
    OMX_ERRORTYPE (*EmptyBufferDone)(
        OMX_IN OMX_HANDLETYPE hComponent,
        OMX_IN OMX_PTR pAppData,
        OMX_IN OMX_BUFFERHEADERTYPE* pBuffer);

    /** The FillBufferDone method is used to return filled buffers from an
        output port back to the application for emptying and then reuse.
        This is a blocking call so the application should not attempt to
        empty the buffers during this call, but should queue the buffers
        and empty them in another thread.  There is no error return, so
        the application shall handle any errors generated internally.  The
        application shall also update the buffer header to indicate the
        number of bytes placed into the buffer.

        The application should return from this call within 5 msec.

        @param hComponent
            handle of the component to access.  This is the component
            handle returned by the call to the GetHandle function.
        @param pAppData
            pointer to an application defined value that was provided in the
            pAppData parameter to the OMX_GetHandle method for the component.
            This application defined value is provided so that the application
            can have a component specific context when receiving the callback.
        @param pBuffer
            pointer to an OMX_BUFFERHEADERTYPE structure allocated with UseBuffer
            or AllocateBuffer indicating the buffer that was filled.
        @ingroup buf
     */
    OMX_ERRORTYPE (*FillBufferDone)(
        OMX_OUT OMX_HANDLETYPE hComponent,
        OMX_OUT OMX_PTR pAppData,
        OMX_OUT OMX_BUFFERHEADERTYPE* pBuffer);

} OMX_CALLBACKTYPE;

/** The OMX_BUFFERSUPPLIERTYPE enumeration is used to dictate port supplier
    preference when tunneling between two ports.
    @ingroup tun buf
*/
typedef enum OMX_BUFFERSUPPLIERTYPE
{
    OMX_BufferSupplyUnspecified = 0x0, /**< port supplying the buffers is unspecified,
                                              or don't care */
    OMX_BufferSupplyInput,             /**< input port supplies the buffers */
    OMX_BufferSupplyOutput,            /**< output port supplies the buffers */
    OMX_BufferSupplyKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
    OMX_BufferSupplyVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_BufferSupplyMax = 0x7FFFFFFF
} OMX_BUFFERSUPPLIERTYPE;


/** buffer supplier parameter
 * @ingroup tun
 */
typedef struct OMX_PARAM_BUFFERSUPPLIERTYPE {
    OMX_U32 nSize; /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion; /**< OMX specification version information */
    OMX_U32 nPortIndex; /**< port that this structure applies to */
    OMX_BUFFERSUPPLIERTYPE eBufferSupplier; /**< buffer supplier */
} OMX_PARAM_BUFFERSUPPLIERTYPE;


/**< indicates that buffers received by an input port of a tunnel
     may not modify the data in the buffers
     @ingroup tun
 */
#define OMX_PORTTUNNELFLAG_READONLY 0x00000001


/** The OMX_TUNNELSETUPTYPE structure is used to pass data from an output
    port to an input port as part the two ComponentTunnelRequest calls
    resulting from a OMX_SetupTunnel call from the IL Client.
    @ingroup tun
 */
typedef struct OMX_TUNNELSETUPTYPE
{
    OMX_U32 nTunnelFlags;             /**< bit flags for tunneling */
    OMX_BUFFERSUPPLIERTYPE eSupplier; /**< supplier preference */
} OMX_TUNNELSETUPTYPE;

/* OMX Component headers is included to enable the core to use
   macros for functions into the component for OMX release 1.0.
   Developers should not access any structures or data from within
   the component header directly */
/* TO BE REMOVED - #include <OMX_Component.h> */

/** GetComponentVersion will return information about the component.
    This is a blocking call.  This macro will go directly from the
    application to the component (via a core macro).  The
    component will return from this call within 5 msec.
    @param [in] hComponent
        handle of component to execute the command
    @param [out] pComponentName
        pointer to an empty string of length 128 bytes.  The component
        will write its name into this string.  The name will be
        terminated by a single zero byte.  The name of a component will
        be 127 bytes or less to leave room for the trailing zero byte.
        An example of a valid component name is "OMX.ABC.ChannelMixer\0".
    @param [out] pComponentVersion
        pointer to an OMX Version structure that the component will fill
        in.  The component will fill in a value that indicates the
        component version.  NOTE: the component version is NOT the same
        as the OMX Specification version (found in all structures).  The
        component version is defined by the vendor of the component and
        its value is entirely up to the component vendor.
    @param [out] pSpecVersion
        pointer to an OMX Version structure that the component will fill
        in.  The SpecVersion is the version of the specification that the
        component was built against.  Please note that this value may or
        may not match the structure's version.  For example, if the
        component was built against the 2.0 specification, but the
        application (which creates the structure is built against the
        1.0 specification the versions would be different.
    @param [out] pComponentUUID
        pointer to the UUID of the component which will be filled in by
        the component.  The UUID is a unique identifier that is set at
        RUN time for the component and is unique to each instantion of
        the component.
    @return OMX_ERRORTYPE
        If the command successfully executes, the return code will be
        OMX_ErrorNone.  Otherwise the appropriate OMX error will be returned.
    @ingroup comp
 */
#define OMX_GetComponentVersion(                            \
        hComponent,                                         \
        pComponentName,                                     \
        pComponentVersion,                                  \
        pSpecVersion,                                       \
        pComponentUUID)                                     \
    ((OMX_COMPONENTTYPE*)hComponent)->GetComponentVersion(  \
        hComponent,                                         \
        pComponentName,                                     \
        pComponentVersion,                                  \
        pSpecVersion,                                       \
        pComponentUUID)                 /* Macro End */


/** Send a command to the component.  This call is a non-blocking call.
    The component should check the parameters and then queue the command
    to the component thread to be executed.  The component thread shall
    send the EventHandler() callback at the conclusion of the command.
    This macro will go directly from the application to the component (via
    a core macro).  The component will return from this call within 5 msec.

    When the command is "OMX_CommandStateSet" the component will queue a
    state transition to the new state idenfied in nParam.

    When the command is "OMX_CommandFlush", to flush a port's buffer queues,
    the command will force the component to return all buffers NOT CURRENTLY
    BEING PROCESSED to the application, in the order in which the buffers
    were received.

    When the command is "OMX_CommandPortDisable" or
    "OMX_CommandPortEnable", the component's port (given by the value of
    nParam) will be stopped or restarted.

    When the command "OMX_CommandMarkBuffer" is used to mark a buffer, the
    pCmdData will point to a OMX_MARKTYPE structure containing the component
    handle of the component to examine the buffer chain for the mark.  nParam1
    contains the index of the port on which the buffer mark is applied.

    Specification text for more details.

    @param [in] hComponent
        handle of component to execute the command
    @param [in] Cmd
        Command for the component to execute
    @param [in] nParam
        Parameter for the command to be executed.  When Cmd has the value
        OMX_CommandStateSet, value is a member of OMX_STATETYPE.  When Cmd has
        the value OMX_CommandFlush, value of nParam indicates which port(s)
        to flush. -1 is used to flush all ports a single port index will
        only flush that port.  When Cmd has the value "OMX_CommandPortDisable"
        or "OMX_CommandPortEnable", the component's port is given by
        the value of nParam.  When Cmd has the value "OMX_CommandMarkBuffer"
        the components pot is given by the value of nParam.
    @param [in] pCmdData
        Parameter pointing to the OMX_MARKTYPE structure when Cmd has the value
        "OMX_CommandMarkBuffer".
    @return OMX_ERRORTYPE
        If the command successfully executes, the return code will be
        OMX_ErrorNone.  Otherwise the appropriate OMX error will be returned.
    @ingroup comp
 */
#define OMX_SendCommand(                                    \
         hComponent,                                        \
         Cmd,                                               \
         nParam,                                            \
         pCmdData)                                          \
     ((OMX_COMPONENTTYPE*)hComponent)->SendCommand(         \
         hComponent,                                        \
         Cmd,                                               \
         nParam,                                            \
         pCmdData)                          /* Macro End */


/** The OMX_GetParameter macro will get one of the current parameter
    settings from the component.  This macro cannot only be invoked when
    the component is in the OMX_StateInvalid state.  The nParamIndex
    parameter is used to indicate which structure is being requested from
    the component.  The application shall allocate the correct structure
    and shall fill in the structure size and version information before
    invoking this macro.  When the parameter applies to a port, the
    caller shall fill in the appropriate nPortIndex value indicating the
    port on which the parameter applies. If the component has not had
    any settings changed, then the component should return a set of
    valid DEFAULT  parameters for the component.  This is a blocking
    call.

    The component should return from this call within 20 msec.

    @param [in] hComponent
        Handle of the component to be accessed.  This is the component
        handle returned by the call to the OMX_GetHandle function.
    @param [in] nParamIndex
        Index of the structure to be filled.  This value is from the
        OMX_INDEXTYPE enumeration.
    @param [in,out] pComponentParameterStructure
        Pointer to application allocated structure to be filled by the
        component.
    @return OMX_ERRORTYPE
        If the command successfully executes, the return code will be
        OMX_ErrorNone.  Otherwise the appropriate OMX error will be returned.
    @ingroup comp
 */
#define OMX_GetParameter(                                   \
        hComponent,                                         \
        nParamIndex,                                        \
        pComponentParameterStructure)                        \
    ((OMX_COMPONENTTYPE*)hComponent)->GetParameter(         \
        hComponent,                                         \
        nParamIndex,                                        \
        pComponentParameterStructure)    /* Macro End */


/** The OMX_SetParameter macro will send an initialization parameter
    structure to a component.  Each structure shall be sent one at a time,
    in a separate invocation of the macro.  This macro can only be
    invoked when the component is in the OMX_StateLoaded state, or the
    port is disabled (when the parameter applies to a port). The
    nParamIndex parameter is used to indicate which structure is being
    passed to the component.  The application shall allocate the
    correct structure and shall fill in the structure size and version
    information (as well as the actual data) before invoking this macro.
    The application is free to dispose of this structure after the call
    as the component is required to copy any data it shall retain.  This
    is a blocking call.

    The component should return from this call within 20 msec.

    @param [in] hComponent
        Handle of the component to be accessed.  This is the component
        handle returned by the call to the OMX_GetHandle function.
    @param [in] nIndex
        Index of the structure to be sent.  This value is from the
        OMX_INDEXTYPE enumeration.
    @param [in] pComponentParameterStructure
        pointer to application allocated structure to be used for
        initialization by the component.
    @return OMX_ERRORTYPE
        If the command successfully executes, the return code will be
        OMX_ErrorNone.  Otherwise the appropriate OMX error will be returned.
    @ingroup comp
 */
#define OMX_SetParameter(                                   \
        hComponent,                                         \
        nParamIndex,                                        \
        pComponentParameterStructure)                        \
    ((OMX_COMPONENTTYPE*)hComponent)->SetParameter(         \
        hComponent,                                         \
        nParamIndex,                                        \
        pComponentParameterStructure)    /* Macro End */


/** The OMX_GetConfig macro will get one of the configuration structures
    from a component.  This macro can be invoked anytime after the
    component has been loaded.  The nParamIndex call parameter is used to
    indicate which structure is being requested from the component.  The
    application shall allocate the correct structure and shall fill in the
    structure size and version information before invoking this macro.
    If the component has not had this configuration parameter sent before,
    then the component should return a set of valid DEFAULT values for the
    component.  This is a blocking call.

    The component should return from this call within 5 msec.

    @param [in] hComponent
        Handle of the component to be accessed.  This is the component
        handle returned by the call to the OMX_GetHandle function.
    @param [in] nIndex
        Index of the structure to be filled.  This value is from the
        OMX_INDEXTYPE enumeration.
    @param [in,out] pComponentConfigStructure
        pointer to application allocated structure to be filled by the
        component.
    @return OMX_ERRORTYPE
        If the command successfully executes, the return code will be
        OMX_ErrorNone.  Otherwise the appropriate OMX error will be returned.
    @ingroup comp
*/
#define OMX_GetConfig(                                      \
        hComponent,                                         \
        nConfigIndex,                                       \
        pComponentConfigStructure)                           \
    ((OMX_COMPONENTTYPE*)hComponent)->GetConfig(            \
        hComponent,                                         \
        nConfigIndex,                                       \
        pComponentConfigStructure)       /* Macro End */


/** The OMX_SetConfig macro will send one of the configuration
    structures to a component.  Each structure shall be sent one at a time,
    each in a separate invocation of the macro.  This macro can be invoked
    anytime after the component has been loaded.  The application shall
    allocate the correct structure and shall fill in the structure size
    and version information (as well as the actual data) before invoking
    this macro.  The application is free to dispose of this structure after
    the call as the component is required to copy any data it shall retain.
    This is a blocking call.

    The component should return from this call within 5 msec.

    @param [in] hComponent
        Handle of the component to be accessed.  This is the component
        handle returned by the call to the OMX_GetHandle function.
    @param [in] nConfigIndex
        Index of the structure to be sent.  This value is from the
        OMX_INDEXTYPE enumeration above.
    @param [in] pComponentConfigStructure
        pointer to application allocated structure to be used for
        initialization by the component.
    @return OMX_ERRORTYPE
        If the command successfully executes, the return code will be
        OMX_ErrorNone.  Otherwise the appropriate OMX error will be returned.
    @ingroup comp
 */
#define OMX_SetConfig(                                      \
        hComponent,                                         \
        nConfigIndex,                                       \
        pComponentConfigStructure)                           \
    ((OMX_COMPONENTTYPE*)hComponent)->SetConfig(            \
        hComponent,                                         \
        nConfigIndex,                                       \
        pComponentConfigStructure)       /* Macro End */


/** The OMX_GetExtensionIndex macro will invoke a component to translate
    a vendor specific configuration or parameter string into an OMX
    structure index.  There is no requirement for the vendor to support
    this command for the indexes already found in the OMX_INDEXTYPE
    enumeration (this is done to save space in small components).  The
    component shall support all vendor supplied extension indexes not found
    in the master OMX_INDEXTYPE enumeration.  This is a blocking call.

    The component should return from this call within 5 msec.

    @param [in] hComponent
        Handle of the component to be accessed.  This is the component
        handle returned by the call to the GetHandle function.
    @param [in] cParameterName
        OMX_STRING that shall be less than 128 characters long including
        the trailing null byte.  This is the string that will get
        translated by the component into a configuration index.
    @param [out] pIndexType
        a pointer to a OMX_INDEXTYPE to receive the index value.
    @return OMX_ERRORTYPE
        If the command successfully executes, the return code will be
        OMX_ErrorNone.  Otherwise the appropriate OMX error will be returned.
    @ingroup comp
 */
#define OMX_GetExtensionIndex(                              \
        hComponent,                                         \
        cParameterName,                                     \
        pIndexType)                                         \
    ((OMX_COMPONENTTYPE*)hComponent)->GetExtensionIndex(    \
        hComponent,                                         \
        cParameterName,                                     \
        pIndexType)                     /* Macro End */


/** The OMX_GetState macro will invoke the component to get the current
    state of the component and place the state value into the location
    pointed to by pState.

    The component should return from this call within 5 msec.

    @param [in] hComponent
        Handle of the component to be accessed.  This is the component
        handle returned by the call to the OMX_GetHandle function.
    @param [out] pState
        pointer to the location to receive the state.  The value returned
        is one of the OMX_STATETYPE members
    @return OMX_ERRORTYPE
        If the command successfully executes, the return code will be
        OMX_ErrorNone.  Otherwise the appropriate OMX error will be returned.
    @ingroup comp
 */
#define OMX_GetState(                                       \
        hComponent,                                         \
        pState)                                             \
    ((OMX_COMPONENTTYPE*)hComponent)->GetState(             \
        hComponent,                                         \
        pState)                         /* Macro End */


/** The OMX_UseBuffer macro will request that the component use
    a buffer (and allocate its own buffer header) already allocated
    by another component, or by the IL Client. This is a blocking
    call.

    The component should return from this call within 20 msec.

    @param [in] hComponent
        Handle of the component to be accessed.  This is the component
        handle returned by the call to the OMX_GetHandle function.
    @param [out] ppBuffer
        pointer to an OMX_BUFFERHEADERTYPE structure used to receive the
        pointer to the buffer header
    @return OMX_ERRORTYPE
        If the command successfully executes, the return code will be
        OMX_ErrorNone.  Otherwise the appropriate OMX error will be returned.
    @ingroup comp buf
 */

#define OMX_UseBuffer(                                      \
           hComponent,                                      \
           ppBufferHdr,                                     \
           nPortIndex,                                      \
           pAppPrivate,                                     \
           nSizeBytes,                                      \
           pBuffer)                                         \
    ((OMX_COMPONENTTYPE*)hComponent)->UseBuffer(            \
           hComponent,                                      \
           ppBufferHdr,                                     \
           nPortIndex,                                      \
           pAppPrivate,                                     \
           nSizeBytes,                                      \
           pBuffer)


/** The OMX_AllocateBuffer macro will request that the component allocate
    a new buffer and buffer header.  The component will allocate the
    buffer and the buffer header and return a pointer to the buffer
    header.  This is a blocking call.

    The component should return from this call within 5 msec.

    @param [in] hComponent
        Handle of the component to be accessed.  This is the component
        handle returned by the call to the OMX_GetHandle function.
    @param [out] ppBuffer
        pointer to an OMX_BUFFERHEADERTYPE structure used to receive
        the pointer to the buffer header
    @param [in] nPortIndex
        nPortIndex is used to select the port on the component the buffer will
        be used with.  The port can be found by using the nPortIndex
        value as an index into the Port Definition array of the component.
    @param [in] pAppPrivate
        pAppPrivate is used to initialize the pAppPrivate member of the
        buffer header structure.
    @param [in] nSizeBytes
        size of the buffer to allocate.  Used when bAllocateNew is true.
    @return OMX_ERRORTYPE
        If the command successfully executes, the return code will be
        OMX_ErrorNone.  Otherwise the appropriate OMX error will be returned.
    @ingroup comp buf
 */
#define OMX_AllocateBuffer(                                 \
        hComponent,                                         \
        ppBuffer,                                           \
        nPortIndex,                                         \
        pAppPrivate,                                        \
        nSizeBytes)                                         \
    ((OMX_COMPONENTTYPE*)hComponent)->AllocateBuffer(       \
        hComponent,                                         \
        ppBuffer,                                           \
        nPortIndex,                                         \
        pAppPrivate,                                        \
        nSizeBytes)                     /* Macro End */


/** The OMX_FreeBuffer macro will release a buffer header from the component
    which was allocated using either OMX_AllocateBuffer or OMX_UseBuffer. If
    the component allocated the buffer (see the OMX_UseBuffer macro) then
    the component shall free the buffer and buffer header. This is a
    blocking call.

    The component should return from this call within 20 msec.

    @param [in] hComponent
        Handle of the component to be accessed.  This is the component
        handle returned by the call to the OMX_GetHandle function.
    @param [in] nPortIndex
        nPortIndex is used to select the port on the component the buffer will
        be used with.
    @param [in] pBuffer
        pointer to an OMX_BUFFERHEADERTYPE structure allocated with UseBuffer
        or AllocateBuffer.
    @return OMX_ERRORTYPE
        If the command successfully executes, the return code will be
        OMX_ErrorNone.  Otherwise the appropriate OMX error will be returned.
    @ingroup comp buf
 */
#define OMX_FreeBuffer(                                     \
        hComponent,                                         \
        nPortIndex,                                         \
        pBuffer)                                            \
    ((OMX_COMPONENTTYPE*)hComponent)->FreeBuffer(           \
        hComponent,                                         \
        nPortIndex,                                         \
        pBuffer)                        /* Macro End */


/** The OMX_EmptyThisBuffer macro will send a buffer full of data to an
    input port of a component.  The buffer will be emptied by the component
    and returned to the application via the EmptyBufferDone call back.
    This is a non-blocking call in that the component will record the buffer
    and return immediately and then empty the buffer, later, at the proper
    time.  As expected, this macro may be invoked only while the component
    is in the OMX_StateExecuting.  If nPortIndex does not specify an input
    port, the component shall return an error.

    The component should return from this call within 5 msec.

    @param [in] hComponent
        Handle of the component to be accessed.  This is the component
        handle returned by the call to the OMX_GetHandle function.
    @param [in] pBuffer
        pointer to an OMX_BUFFERHEADERTYPE structure allocated with UseBuffer
        or AllocateBuffer.
    @return OMX_ERRORTYPE
        If the command successfully executes, the return code will be
        OMX_ErrorNone.  Otherwise the appropriate OMX error will be returned.
    @ingroup comp buf
 */
#define OMX_EmptyThisBuffer(                                \
        hComponent,                                         \
        pBuffer)                                            \
    ((OMX_COMPONENTTYPE*)hComponent)->EmptyThisBuffer(      \
        hComponent,                                         \
        pBuffer)                        /* Macro End */


/** The OMX_FillThisBuffer macro will send an empty buffer to an
    output port of a component.  The buffer will be filled by the component
    and returned to the application via the FillBufferDone call back.
    This is a non-blocking call in that the component will record the buffer
    and return immediately and then fill the buffer, later, at the proper
    time.  As expected, this macro may be invoked only while the component
    is in the OMX_ExecutingState.  If nPortIndex does not specify an output
    port, the component shall return an error.

    The component should return from this call within 5 msec.

    @param [in] hComponent
        Handle of the component to be accessed.  This is the component
        handle returned by the call to the OMX_GetHandle function.
    @param [in] pBuffer
        pointer to an OMX_BUFFERHEADERTYPE structure allocated with UseBuffer
        or AllocateBuffer.
    @return OMX_ERRORTYPE
        If the command successfully executes, the return code will be
        OMX_ErrorNone.  Otherwise the appropriate OMX error will be returned.
    @ingroup comp buf
 */
#define OMX_FillThisBuffer(                                 \
        hComponent,                                         \
        pBuffer)                                            \
    ((OMX_COMPONENTTYPE*)hComponent)->FillThisBuffer(       \
        hComponent,                                         \
        pBuffer)                        /* Macro End */



/** The OMX_UseEGLImage macro will request that the component use
    a EGLImage provided by EGL (and allocate its own buffer header)
    This is a blocking call.

    The component should return from this call within 20 msec.

    @param [in] hComponent
        Handle of the component to be accessed.  This is the component
        handle returned by the call to the OMX_GetHandle function.
    @param [out] ppBuffer
        pointer to an OMX_BUFFERHEADERTYPE structure used to receive the
        pointer to the buffer header.  Note that the memory location used
        for this buffer is NOT visible to the IL Client.
    @param [in] nPortIndex
        nPortIndex is used to select the port on the component the buffer will
        be used with.  The port can be found by using the nPortIndex
        value as an index into the Port Definition array of the component.
    @param [in] pAppPrivate
        pAppPrivate is used to initialize the pAppPrivate member of the
        buffer header structure.
    @param [in] eglImage
        eglImage contains the handle of the EGLImage to use as a buffer on the
        specified port.  The component is expected to validate properties of
        the EGLImage against the configuration of the port to ensure the component
        can use the EGLImage as a buffer.
    @return OMX_ERRORTYPE
        If the command successfully executes, the return code will be
        OMX_ErrorNone.  Otherwise the appropriate OMX error will be returned.
    @ingroup comp buf
 */
#define OMX_UseEGLImage(                                    \
           hComponent,                                      \
           ppBufferHdr,                                     \
           nPortIndex,                                      \
           pAppPrivate,                                     \
           eglImage)                                        \
    ((OMX_COMPONENTTYPE*)hComponent)->UseEGLImage(          \
           hComponent,                                      \
           ppBufferHdr,                                     \
           nPortIndex,                                      \
           pAppPrivate,                                     \
           eglImage)

/** The OMX_Init method is used to initialize the OMX core.  It shall be the
    first call made into OMX and it should only be executed one time without
    an interviening OMX_Deinit call.

    The core should return from this call within 20 msec.

    @return OMX_ERRORTYPE
        If the command successfully executes, the return code will be
        OMX_ErrorNone.  Otherwise the appropriate OMX error will be returned.
    @ingroup core
 */
OMX_API OMX_ERRORTYPE OMX_APIENTRY OMX_Init(void);


/** The OMX_Deinit method is used to deinitialize the OMX core.  It shall be
    the last call made into OMX. In the event that the core determines that
    thare are components loaded when this call is made, the core may return
    with an error rather than try to unload the components.

    The core should return from this call within 20 msec.

    @return OMX_ERRORTYPE
        If the command successfully executes, the return code will be
        OMX_ErrorNone.  Otherwise the appropriate OMX error will be returned.
    @ingroup core
 */
OMX_API OMX_ERRORTYPE OMX_APIENTRY OMX_Deinit(void);


/** The OMX_ComponentNameEnum method will enumerate through all the names of
    recognised valid components in the system. This function is provided
    as a means to detect all the components in the system run-time. There is
    no strict ordering to the enumeration order of component names, although
    each name will only be enumerated once.  If the OMX core supports run-time
    installation of new components, it is only requried to detect newly
    installed components when the first call to enumerate component names
    is made (i.e. when nIndex is 0x0).

    The core should return from this call in 20 msec.

    @param [out] cComponentName
        pointer to a null terminated string with the component name.  The
        names of the components are strings less than 127 bytes in length
        plus the trailing null for a maximum size of 128 bytes.  An example
        of a valid component name is "OMX.TI.AUDIO.DSP.MIXER\0".  Names are
        assigned by the vendor, but shall start with "OMX." and then have
        the Vendor designation next.
    @param [in] nNameLength
        number of characters in the cComponentName string.  With all
        component name strings restricted to less than 128 characters
        (including the trailing null) it is recomended that the caller
        provide a input string for the cComponentName of 128 characters.
    @param [in] nIndex
        number containing the enumeration index for the component.
        Multiple calls to OMX_ComponentNameEnum with increasing values
        of nIndex will enumerate through the component names in the
        system until OMX_ErrorNoMore is returned.  The value of nIndex
        is 0 to (N-1), where N is the number of valid installed components
        in the system.
    @return OMX_ERRORTYPE
        If the command successfully executes, the return code will be
        OMX_ErrorNone.  When the value of nIndex exceeds the number of
        components in the system minus 1, OMX_ErrorNoMore will be
        returned. Otherwise the appropriate OMX error will be returned.
    @ingroup core
 */
OMX_API OMX_ERRORTYPE OMX_APIENTRY OMX_ComponentNameEnum(
    OMX_OUT OMX_STRING cComponentName,
    OMX_IN  OMX_U32 nNameLength,
    OMX_IN  OMX_U32 nIndex);


/** The OMX_GetHandle method will locate the component specified by the
    component name given, load that component into memory and then invoke
    the component's methods to create an instance of the component.

    The core should return from this call within 20 msec.

    @param [out] pHandle
        pointer to an OMX_HANDLETYPE pointer to be filled in by this method.
    @param [in] cComponentName
        pointer to a null terminated string with the component name.  The
        names of the components are strings less than 127 bytes in length
        plus the trailing null for a maximum size of 128 bytes.  An example
        of a valid component name is "OMX.TI.AUDIO.DSP.MIXER\0".  Names are
        assigned by the vendor, but shall start with "OMX." and then have
        the Vendor designation next.
    @param [in] pAppData
        pointer to an application defined value that will be returned
        during callbacks so that the application can identify the source
        of the callback.
    @param [in] pCallBacks
        pointer to a OMX_CALLBACKTYPE structure that will be passed to the
        component to initialize it with.
    @return OMX_ERRORTYPE
        If the command successfully executes, the return code will be
        OMX_ErrorNone.  Otherwise the appropriate OMX error will be returned.
    @ingroup core
 */
OMX_API OMX_ERRORTYPE OMX_APIENTRY OMX_GetHandle(
    OMX_OUT OMX_HANDLETYPE* pHandle,
    OMX_IN  OMX_STRING cComponentName,
    OMX_IN  OMX_PTR pAppData,
    OMX_IN  OMX_CALLBACKTYPE* pCallBacks);


/** The OMX_FreeHandle method will free a handle allocated by the OMX_GetHandle
    method.  If the component reference count goes to zero, the component will
    be unloaded from memory.

    The core should return from this call within 20 msec when the component is
    in the OMX_StateLoaded state.

    @param [in] hComponent
        Handle of the component to be accessed.  This is the component
        handle returned by the call to the GetHandle function.
    @return OMX_ERRORTYPE
        If the command successfully executes, the return code will be
        OMX_ErrorNone.  Otherwise the appropriate OMX error will be returned.
    @ingroup core
 */
OMX_API OMX_ERRORTYPE OMX_APIENTRY OMX_FreeHandle(
    OMX_IN  OMX_HANDLETYPE hComponent);



/** The OMX_SetupTunnel method will handle the necessary calls to the components
    to setup the specified tunnel the two components.  NOTE: This is
    an actual method (not a #define macro).  This method will make calls into
    the component ComponentTunnelRequest method to do the actual tunnel
    connection.

    The ComponentTunnelRequest method on both components will be called.
    This method shall not be called unless the component is in the
    OMX_StateLoaded state except when the ports used for the tunnel are
    disabled. In this case, the component may be in the OMX_StateExecuting,
    OMX_StatePause, or OMX_StateIdle states.

    The core should return from this call within 20 msec.

    @param [in] hOutput
        Handle of the component to be accessed.  Also this is the handle
        of the component whose port, specified in the nPortOutput parameter
        will be used the source for the tunnel. This is the component handle
        returned by the call to the OMX_GetHandle function.  There is a
        requirement that hOutput be the source for the data when
        tunelling (i.e. nPortOutput is an output port).  If 0x0, the component
        specified in hInput will have it's port specified in nPortInput
        setup for communication with the application / IL client.
    @param [in] nPortOutput
        nPortOutput is used to select the source port on component to be
        used in the tunnel.
    @param [in] hInput
        This is the component to setup the tunnel with. This is the handle
        of the component whose port, specified in the nPortInput parameter
        will be used the destination for the tunnel. This is the component handle
        returned by the call to the OMX_GetHandle function.  There is a
        requirement that hInput be the destination for the data when
        tunelling (i.e. nPortInut is an input port).   If 0x0, the component
        specified in hOutput will have it's port specified in nPortPOutput
        setup for communication with the application / IL client.
    @param [in] nPortInput
        nPortInput is used to select the destination port on component to be
        used in the tunnel.
    @return OMX_ERRORTYPE
        If the command successfully executes, the return code will be
        OMX_ErrorNone.  Otherwise the appropriate OMX error will be returned.
        When OMX_ErrorNotImplemented is returned, one or both components is
        a non-interop component and does not support tunneling.

        On failure, the ports of both components are setup for communication
        with the application / IL Client.
    @ingroup core tun
 */
OMX_API OMX_ERRORTYPE OMX_APIENTRY OMX_SetupTunnel(
    OMX_IN  OMX_HANDLETYPE hOutput,
    OMX_IN  OMX_U32 nPortOutput,
    OMX_IN  OMX_HANDLETYPE hInput,
    OMX_IN  OMX_U32 nPortInput);

/** @ingroup cp */
OMX_API OMX_ERRORTYPE   OMX_GetContentPipe(
    OMX_OUT OMX_HANDLETYPE *hPipe,
    OMX_IN OMX_STRING szURI);

/** The OMX_GetComponentsOfRole method will return the number of components that support the given
    role and (if the compNames field is non-NULL) the names of those components. The call will fail if
    an insufficiently sized array of names is supplied. To ensure the array is sufficiently sized the
    client should:
        * first call this function with the compNames field NULL to determine the number of component names
        * second call this function with the compNames field pointing to an array of names allocated
          according to the number returned by the first call.

    The core should return from this call within 5 msec.

    @param [in] role
        This is generic standard component name consisting only of component class
        name and the type within that class (e.g. 'audio_decoder.aac').
    @param [inout] pNumComps
        This is used both as input and output.

        If compNames is NULL, the input is ignored and the output specifies how many components support
        the given role.

        If compNames is not NULL, on input it bounds the size of the input structure and
        on output, it specifies the number of components string names listed within the compNames parameter.
    @param [inout] compNames
        If NULL this field is ignored. If non-NULL this points to an array of 128-byte strings which accepts
        a list of the names of all physical components that implement the specified standard component name.
        Each name is NULL terminated. numComps indicates the number of names.
    @ingroup core
 */
OMX_API OMX_ERRORTYPE OMX_GetComponentsOfRole (
    OMX_IN      OMX_STRING role,
    OMX_INOUT   OMX_U32 *pNumComps,
    OMX_INOUT   OMX_U8  **compNames);

/** The OMX_GetRolesOfComponent method will return the number of roles supported by the given
    component and (if the roles field is non-NULL) the names of those roles. The call will fail if
    an insufficiently sized array of names is supplied. To ensure the array is sufficiently sized the
    client should:
        * first call this function with the roles field NULL to determine the number of role names
        * second call this function with the roles field pointing to an array of names allocated
          according to the number returned by the first call.

    The core should return from this call within 5 msec.

    @param [in] compName
        This is the name of the component being queried about.
    @param [inout] pNumRoles
        This is used both as input and output.

        If roles is NULL, the input is ignored and the output specifies how many roles the component supports.

        If compNames is not NULL, on input it bounds the size of the input structure and
        on output, it specifies the number of roles string names listed within the roles parameter.
    @param [out] roles
        If NULL this field is ignored. If non-NULL this points to an array of 128-byte strings
        which accepts a list of the names of all standard components roles implemented on the
        specified component name. numComps indicates the number of names.
    @ingroup core
 */
OMX_API OMX_ERRORTYPE OMX_GetRolesOfComponent (
    OMX_IN      OMX_STRING compName,
    OMX_INOUT   OMX_U32 *pNumRoles,
    OMX_OUT     OMX_U8 **roles);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
/* File EOF */

