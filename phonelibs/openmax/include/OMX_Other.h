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

/** @file OMX_Other.h - OpenMax IL version 1.1.2
 *  The structures needed by Other components to exchange
 *  parameters and configuration data with the components.
 */

#ifndef OMX_Other_h
#define OMX_Other_h

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */


/* Each OMX header must include all required header files to allow the
 *  header to compile without errors.  The includes below are required
 *  for this header file to compile successfully
 */

#include <OMX_Core.h>


/**
 * Enumeration of possible data types which match to multiple domains or no
 * domain at all.  For types which are vendor specific, a value above
 * OMX_OTHER_VENDORTSTART should be used.
 */
typedef enum OMX_OTHER_FORMATTYPE {
    OMX_OTHER_FormatTime = 0, /**< Transmission of various timestamps, elapsed time,
                                   time deltas, etc */
    OMX_OTHER_FormatPower,    /**< Perhaps used for enabling/disabling power
                                   management, setting clocks? */
    OMX_OTHER_FormatStats,    /**< Could be things such as frame rate, frames
                                   dropped, etc */
    OMX_OTHER_FormatBinary,   /**< Arbitrary binary data */
    OMX_OTHER_FormatVendorReserved = 1000, /**< Starting value for vendor specific
                                                formats */

    OMX_OTHER_FormatKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
    OMX_OTHER_FormatVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_OTHER_FormatMax = 0x7FFFFFFF
} OMX_OTHER_FORMATTYPE;

/**
 * Enumeration of seek modes.
 */
typedef enum OMX_TIME_SEEKMODETYPE {
    OMX_TIME_SeekModeFast = 0, /**< Prefer seeking to an approximation
                                * of the requested seek position over
                                * the actual seek position if it
                                * results in a faster seek. */
    OMX_TIME_SeekModeAccurate, /**< Prefer seeking to the actual seek
                                * position over an approximation
                                * of the requested seek position even
                                * if it results in a slower seek. */
    OMX_TIME_SeekModeKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
    OMX_TIME_SeekModeVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_TIME_SeekModeMax = 0x7FFFFFFF
} OMX_TIME_SEEKMODETYPE;

/* Structure representing the seekmode of the component */
typedef struct OMX_TIME_CONFIG_SEEKMODETYPE {
    OMX_U32 nSize;                  /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion;       /**< OMX specification version information */
    OMX_TIME_SEEKMODETYPE eType;    /**< The seek mode */
} OMX_TIME_CONFIG_SEEKMODETYPE;

/** Structure representing a time stamp used with the following configs
 * on the Clock Component (CC):
 *
 * OMX_IndexConfigTimeCurrentWallTime: query of the CC’s current wall
 *     time
 * OMX_IndexConfigTimeCurrentMediaTime: query of the CC’s current media
 *     time
 * OMX_IndexConfigTimeCurrentAudioReference and
 * OMX_IndexConfigTimeCurrentVideoReference: audio/video reference
 *     clock sending SC its reference time
 * OMX_IndexConfigTimeClientStartTime: a Clock Component client sends
 *     this structure to the Clock Component via a SetConfig on its
 *     client port when it receives a buffer with
 *     OMX_BUFFERFLAG_STARTTIME set. It must use the timestamp
 *     specified by that buffer for nStartTimestamp.
 *
 * It’s also used with the following config on components in general:
 *
 * OMX_IndexConfigTimePosition: IL client querying component position
 * (GetConfig) or commanding a component to seek to the given location
 * (SetConfig)
 */
typedef struct OMX_TIME_CONFIG_TIMESTAMPTYPE {
    OMX_U32 nSize;               /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion;    /**< OMX specification version
                                  *   information */
    OMX_U32 nPortIndex;     /**< port that this structure applies to */
    OMX_TICKS nTimestamp;  	     /**< timestamp .*/
} OMX_TIME_CONFIG_TIMESTAMPTYPE;

/** Enumeration of possible reference clocks to the media time. */
typedef enum OMX_TIME_UPDATETYPE {
      OMX_TIME_UpdateRequestFulfillment,    /**< Update is the fulfillment of a media time request. */
      OMX_TIME_UpdateScaleChanged,	        /**< Update was generated because the scale chagned. */
      OMX_TIME_UpdateClockStateChanged,     /**< Update was generated because the clock state changed. */
      OMX_TIME_UpdateKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
      OMX_TIME_UpdateVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
      OMX_TIME_UpdateMax = 0x7FFFFFFF
} OMX_TIME_UPDATETYPE;

/** Enumeration of possible reference clocks to the media time. */
typedef enum OMX_TIME_REFCLOCKTYPE {
      OMX_TIME_RefClockNone,    /**< Use no references. */
      OMX_TIME_RefClockAudio,	/**< Use references sent through OMX_IndexConfigTimeCurrentAudioReference */
      OMX_TIME_RefClockVideo,   /**< Use references sent through OMX_IndexConfigTimeCurrentVideoReference */
      OMX_TIME_RefClockKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
      OMX_TIME_RefClockVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
      OMX_TIME_RefClockMax = 0x7FFFFFFF
} OMX_TIME_REFCLOCKTYPE;

/** Enumeration of clock states. */
typedef enum OMX_TIME_CLOCKSTATE {
      OMX_TIME_ClockStateRunning,             /**< Clock running. */
      OMX_TIME_ClockStateWaitingForStartTime, /**< Clock waiting until the
                                               *   prescribed clients emit their
                                               *   start time. */
      OMX_TIME_ClockStateStopped,             /**< Clock stopped. */
      OMX_TIME_ClockStateKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
      OMX_TIME_ClockStateVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
      OMX_TIME_ClockStateMax = 0x7FFFFFFF
} OMX_TIME_CLOCKSTATE;

/** Structure representing a media time request to the clock component.
 *
 *  A client component sends this structure to the Clock Component via a SetConfig
 *  on its client port to specify a media timestamp the Clock Component
 *  should emit.  The Clock Component should fulfill the request by sending a
 *  OMX_TIME_MEDIATIMETYPE when its media clock matches the requested
 *  timestamp.
 *
 *  The client may require a media time request be fulfilled slightly
 *  earlier than the media time specified. In this case the client specifies
 *  an offset which is equal to the difference between wall time corresponding
 *  to the requested media time and the wall time when it will be
 *  fulfilled.
 *
 *  A client component may uses these requests and the OMX_TIME_MEDIATIMETYPE to
 *  time events according to timestamps. If a client must perform an operation O at
 *  a time T (e.g. deliver a video frame at its corresponding timestamp), it makes a
 *  media time request at T (perhaps specifying an offset to ensure the request fulfillment
 *  is a little early). When the clock component passes the resulting OMX_TIME_MEDIATIMETYPE
 *  structure back to the client component, the client may perform operation O (perhaps having
 *  to wait a slight amount more time itself as specified by the return values).
 */

typedef struct OMX_TIME_CONFIG_MEDIATIMEREQUESTTYPE {
    OMX_U32 nSize;              /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion;   /**< OMX specification version information */
    OMX_U32 nPortIndex;         /**< port that this structure applies to */
    OMX_PTR pClientPrivate;     /**< Client private data to disabiguate this media time
                                 *   from others (e.g. the number of the frame to deliver).
                                 *   Duplicated in the media time structure that fulfills
                                 *   this request. A value of zero is reserved for time scale
                                 *   updates. */
    OMX_TICKS nMediaTimestamp;  /**< Media timestamp requested.*/
    OMX_TICKS nOffset;          /**< Amount of wall clock time by which this
                                 *   request should be fulfilled early */
} OMX_TIME_CONFIG_MEDIATIMEREQUESTTYPE;

/**< Structure sent from the clock component client either when fulfilling
 *   a media time request or when the time scale has changed.
 *
 *   In the former case the Clock Component fills this structure and times its emission
 *   to a client component (via the client port) according to the corresponding media
 *   time request sent by the client. The Clock Component should time the emission to occur
 *   when the requested timestamp matches the Clock Component's media time but also the
 *   prescribed offset early.
 *
 *   Upon scale changes the clock component clears the nClientPrivate data, sends the current
 *   media time and sets the nScale to the new scale via the client port. It emits a
 *   OMX_TIME_MEDIATIMETYPE to all clients independent of any requests. This allows clients to
 *   alter processing to accomodate scaling. For instance a video component might skip inter-frames
 *   in the case of extreme fastforward. Likewise an audio component might add or remove samples
 *   from an audio frame to scale audio data.
 *
 *   It is expected that some clock components may not be able to fulfill requests
 *   at exactly the prescribed time. This is acceptable so long as the request is
 *   fulfilled at least as early as described and not later. This structure provides
 *   fields the client may use to wait for the remaining time.
 *
 *   The client may use either the nOffset or nWallTimeAtMedia fields to determine the
 *   wall time until the nMediaTimestamp actually occurs. In the latter case the
 *   client can get a more accurate value for offset by getting the current wall
 *   from the cloc component and subtracting it from nWallTimeAtMedia.
 */

typedef struct OMX_TIME_MEDIATIMETYPE {
    OMX_U32 nSize;                  /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion;       /**< OMX specification version information */
    OMX_U32 nClientPrivate;         /**< Client private data to disabiguate this media time
                                     *   from others. Copied from the media time request.
                                     *   A value of zero is reserved for time scale updates. */
    OMX_TIME_UPDATETYPE eUpdateType; /**< Reason for the update */
    OMX_TICKS nMediaTimestamp;      /**< Media time requested. If no media time was
                                     *   requested then this is the current media time. */
    OMX_TICKS nOffset;              /**< Amount of wall clock time by which this
                                     *   request was actually fulfilled early */

    OMX_TICKS nWallTimeAtMediaTime; /**< Wall time corresponding to nMediaTimeStamp.
                                     *   A client may compare this value to current
                                     *   media time obtained from the Clock Component to determine
                                     *   the wall time until the media timestamp is really
                                     *   current. */
    OMX_S32 xScale;                 /**< Current media time scale in Q16 format. */
    OMX_TIME_CLOCKSTATE eState;     /* Seeking Change. Added 7/12.*/
                                    /**< State of the media time. */
} OMX_TIME_MEDIATIMETYPE;

/** Structure representing the current media time scale factor. Applicable only to clock
 *  component, other components see scale changes via OMX_TIME_MEDIATIMETYPE buffers sent via
 *  the clock component client ports. Upon recieving this config the clock component changes
 *  the rate by which the media time increases or decreases effectively implementing trick modes.
 */
typedef struct OMX_TIME_CONFIG_SCALETYPE {
    OMX_U32 nSize;                  /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion;       /**< OMX specification version information */
    OMX_S32 xScale;                 /**< This is a value in Q16 format which is used for
                                     * scaling the media time */
} OMX_TIME_CONFIG_SCALETYPE;

/** Bits used to identify a clock port. Used in OMX_TIME_CONFIG_CLOCKSTATETYPE’s nWaitMask field */
#define OMX_CLOCKPORT0 0x00000001
#define OMX_CLOCKPORT1 0x00000002
#define OMX_CLOCKPORT2 0x00000004
#define OMX_CLOCKPORT3 0x00000008
#define OMX_CLOCKPORT4 0x00000010
#define OMX_CLOCKPORT5 0x00000020
#define OMX_CLOCKPORT6 0x00000040
#define OMX_CLOCKPORT7 0x00000080

/** Structure representing the current mode of the media clock.
 *  IL Client uses this config to change or query the mode of the
 *  media clock of the clock component. Applicable only to clock
 *  component.
 *
 *  On a SetConfig if eState is OMX_TIME_ClockStateRunning media time
 *  starts immediately at the prescribed start time. If
 *  OMX_TIME_ClockStateWaitingForStartTime the Clock Component ignores
 *  the given nStartTime and waits for all clients specified in the
 *  nWaitMask to send starttimes (via
 *  OMX_IndexConfigTimeClientStartTime). The Clock Component then starts
 *  the media clock using the earliest start time supplied. */
typedef struct OMX_TIME_CONFIG_CLOCKSTATETYPE {
    OMX_U32 nSize;              /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion;   /**< OMX specification version
                                 *   information */
    OMX_TIME_CLOCKSTATE eState; /**< State of the media time. */
    OMX_TICKS nStartTime;       /**< Start time of the media time. */
    OMX_TICKS nOffset;          /**< Time to offset the media time by
                                 * (e.g. preroll). Media time will be
                                 * reported to be nOffset ticks earlier.
                                 */
    OMX_U32 nWaitMask;          /**< Mask of OMX_CLOCKPORT values. */
} OMX_TIME_CONFIG_CLOCKSTATETYPE;

/** Structure representing the reference clock currently being used to
 *  compute media time. IL client uses this config to change or query the
 *  clock component's active reference clock */
typedef struct OMX_TIME_CONFIG_ACTIVEREFCLOCKTYPE {
    OMX_U32 nSize;                  /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion;       /**< OMX specification version information */
    OMX_TIME_REFCLOCKTYPE eClock;   /**< Reference clock used to compute media time */
} OMX_TIME_CONFIG_ACTIVEREFCLOCKTYPE;

/** Descriptor for setting specifics of power type.
 *  Note: this structure is listed for backwards compatibility. */
typedef struct OMX_OTHER_CONFIG_POWERTYPE {
    OMX_U32 nSize;            /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion; /**< OMX specification version information */
    OMX_BOOL bEnablePM;       /**< Flag to enable Power Management */
} OMX_OTHER_CONFIG_POWERTYPE;


/** Descriptor for setting specifics of stats type.
 *  Note: this structure is listed for backwards compatibility. */
typedef struct OMX_OTHER_CONFIG_STATSTYPE {
    OMX_U32 nSize;            /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion; /**< OMX specification version information */
    /* what goes here */
} OMX_OTHER_CONFIG_STATSTYPE;


/**
 * The PortDefinition structure is used to define all of the parameters
 * necessary for the compliant component to setup an input or an output other
 * path.
 */
typedef struct OMX_OTHER_PORTDEFINITIONTYPE {
    OMX_OTHER_FORMATTYPE eFormat;  /**< Type of data expected for this channel */
} OMX_OTHER_PORTDEFINITIONTYPE;

/**  Port format parameter.  This structure is used to enumerate
  *  the various data input/output format supported by the port.
  */
typedef struct OMX_OTHER_PARAM_PORTFORMATTYPE {
    OMX_U32 nSize; /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion; /**< OMX specification version information */
    OMX_U32 nPortIndex; /**< Indicates which port to set */
    OMX_U32 nIndex; /**< Indicates the enumeration index for the format from 0x0 to N-1 */
    OMX_OTHER_FORMATTYPE eFormat; /**< Type of data expected for this channel */
} OMX_OTHER_PARAM_PORTFORMATTYPE;

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
/* File EOF */
