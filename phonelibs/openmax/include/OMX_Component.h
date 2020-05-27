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

/** OMX_Component.h - OpenMax IL version 1.1.2
 *  The OMX_Component header file contains the definitions used to define
 *  the public interface of a component.  This header file is intended to
 *  be used by both the application and the component.
 */

#ifndef OMX_Component_h
#define OMX_Component_h

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */



/* Each OMX header must include all required header files to allow the
 *  header to compile without errors.  The includes below are required
 *  for this header file to compile successfully
 */

#include <OMX_Audio.h>
#include <OMX_Video.h>
#include <OMX_Image.h>
#include <OMX_Other.h>

/** @ingroup comp */
typedef enum OMX_PORTDOMAINTYPE {
    OMX_PortDomainAudio,
    OMX_PortDomainVideo,
    OMX_PortDomainImage,
    OMX_PortDomainOther,
    OMX_PortDomainKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
    OMX_PortDomainVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_PortDomainMax = 0x7ffffff
} OMX_PORTDOMAINTYPE;

/** @ingroup comp */
typedef struct OMX_PARAM_PORTDEFINITIONTYPE {
    OMX_U32 nSize;                 /**< Size of the structure in bytes */
    OMX_VERSIONTYPE nVersion;      /**< OMX specification version information */
    OMX_U32 nPortIndex;            /**< Port number the structure applies to */
    OMX_DIRTYPE eDir;              /**< Direction (input or output) of this port */
    OMX_U32 nBufferCountActual;    /**< The actual number of buffers allocated on this port */
    OMX_U32 nBufferCountMin;       /**< The minimum number of buffers this port requires */
    OMX_U32 nBufferSize;           /**< Size, in bytes, for buffers to be used for this channel */
    OMX_BOOL bEnabled;             /**< Ports default to enabled and are enabled/disabled by
                                        OMX_CommandPortEnable/OMX_CommandPortDisable.
                                        When disabled a port is unpopulated. A disabled port
                                        is not populated with buffers on a transition to IDLE. */
    OMX_BOOL bPopulated;           /**< Port is populated with all of its buffers as indicated by
                                        nBufferCountActual. A disabled port is always unpopulated.
                                        An enabled port is populated on a transition to OMX_StateIdle
                                        and unpopulated on a transition to loaded. */
    OMX_PORTDOMAINTYPE eDomain;    /**< Domain of the port. Determines the contents of metadata below. */
    union {
        OMX_AUDIO_PORTDEFINITIONTYPE audio;
        OMX_VIDEO_PORTDEFINITIONTYPE video;
        OMX_IMAGE_PORTDEFINITIONTYPE image;
        OMX_OTHER_PORTDEFINITIONTYPE other;
    } format;
    OMX_BOOL bBuffersContiguous;
    OMX_U32 nBufferAlignment;
} OMX_PARAM_PORTDEFINITIONTYPE;

/** @ingroup comp */
typedef struct OMX_PARAM_U32TYPE {
    OMX_U32 nSize;                    /**< Size of this structure, in Bytes */
    OMX_VERSIONTYPE nVersion;         /**< OMX specification version information */
    OMX_U32 nPortIndex;               /**< port that this structure applies to */
    OMX_U32 nU32;                     /**< U32 value */
} OMX_PARAM_U32TYPE;

/** @ingroup rpm */
typedef enum OMX_SUSPENSIONPOLICYTYPE {
    OMX_SuspensionDisabled, /**< No suspension; v1.0 behavior */
    OMX_SuspensionEnabled,  /**< Suspension allowed */
    OMX_SuspensionPolicyKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
    OMX_SuspensionPolicyStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_SuspensionPolicyMax = 0x7fffffff
} OMX_SUSPENSIONPOLICYTYPE;

/** @ingroup rpm */
typedef struct OMX_PARAM_SUSPENSIONPOLICYTYPE {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_SUSPENSIONPOLICYTYPE ePolicy;
} OMX_PARAM_SUSPENSIONPOLICYTYPE;

/** @ingroup rpm */
typedef enum OMX_SUSPENSIONTYPE {
    OMX_NotSuspended, /**< component is not suspended */
    OMX_Suspended,    /**< component is suspended */
    OMX_SuspensionKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
    OMX_SuspensionVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_SuspendMax = 0x7FFFFFFF
} OMX_SUSPENSIONTYPE;

/** @ingroup rpm */
typedef struct OMX_PARAM_SUSPENSIONTYPE {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_SUSPENSIONTYPE eType;
} OMX_PARAM_SUSPENSIONTYPE ;

typedef struct OMX_CONFIG_BOOLEANTYPE {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_BOOL bEnabled;
} OMX_CONFIG_BOOLEANTYPE;

/* Parameter specifying the content uri to use. */
/** @ingroup cp */
typedef struct OMX_PARAM_CONTENTURITYPE
{
    OMX_U32 nSize;                      /**< size of the structure in bytes, including
                                             actual URI name */
    OMX_VERSIONTYPE nVersion;           /**< OMX specification version information */
    OMX_U8 contentURI[1];               /**< The URI name */
} OMX_PARAM_CONTENTURITYPE;

/* Parameter specifying the pipe to use. */
/** @ingroup cp */
typedef struct OMX_PARAM_CONTENTPIPETYPE
{
    OMX_U32 nSize;              /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion;   /**< OMX specification version information */
    OMX_HANDLETYPE hPipe;       /**< The pipe handle*/
} OMX_PARAM_CONTENTPIPETYPE;

/** @ingroup rpm */
typedef struct OMX_RESOURCECONCEALMENTTYPE {
    OMX_U32 nSize;             /**< size of the structure in bytes */
    OMX_VERSIONTYPE nVersion;  /**< OMX specification version information */
    OMX_BOOL bResourceConcealmentForbidden; /**< disallow the use of resource concealment
                                            methods (like degrading algorithm quality to
                                            lower resource consumption or functional bypass)
                                            on a component as a resolution to resource conflicts. */
} OMX_RESOURCECONCEALMENTTYPE;


/** @ingroup metadata */
typedef enum OMX_METADATACHARSETTYPE {
    OMX_MetadataCharsetUnknown = 0,
    OMX_MetadataCharsetASCII,
    OMX_MetadataCharsetBinary,
    OMX_MetadataCharsetCodePage1252,
    OMX_MetadataCharsetUTF8,
    OMX_MetadataCharsetJavaConformantUTF8,
    OMX_MetadataCharsetUTF7,
    OMX_MetadataCharsetImapUTF7,
    OMX_MetadataCharsetUTF16LE,
    OMX_MetadataCharsetUTF16BE,
    OMX_MetadataCharsetGB12345,
    OMX_MetadataCharsetHZGB2312,
    OMX_MetadataCharsetGB2312,
    OMX_MetadataCharsetGB18030,
    OMX_MetadataCharsetGBK,
    OMX_MetadataCharsetBig5,
    OMX_MetadataCharsetISO88591,
    OMX_MetadataCharsetISO88592,
    OMX_MetadataCharsetISO88593,
    OMX_MetadataCharsetISO88594,
    OMX_MetadataCharsetISO88595,
    OMX_MetadataCharsetISO88596,
    OMX_MetadataCharsetISO88597,
    OMX_MetadataCharsetISO88598,
    OMX_MetadataCharsetISO88599,
    OMX_MetadataCharsetISO885910,
    OMX_MetadataCharsetISO885913,
    OMX_MetadataCharsetISO885914,
    OMX_MetadataCharsetISO885915,
    OMX_MetadataCharsetShiftJIS,
    OMX_MetadataCharsetISO2022JP,
    OMX_MetadataCharsetISO2022JP1,
    OMX_MetadataCharsetISOEUCJP,
    OMX_MetadataCharsetSMS7Bit,
    OMX_MetadataCharsetKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
    OMX_MetadataCharsetVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_MetadataCharsetTypeMax= 0x7FFFFFFF
} OMX_METADATACHARSETTYPE;

/** @ingroup metadata */
typedef enum OMX_METADATASCOPETYPE
{
    OMX_MetadataScopeAllLevels,
    OMX_MetadataScopeTopLevel,
    OMX_MetadataScopePortLevel,
    OMX_MetadataScopeNodeLevel,
    OMX_MetadataScopeKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
    OMX_MetadataScopeVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_MetadataScopeTypeMax = 0x7fffffff
} OMX_METADATASCOPETYPE;

/** @ingroup metadata */
typedef enum OMX_METADATASEARCHMODETYPE
{
    OMX_MetadataSearchValueSizeByIndex,
    OMX_MetadataSearchItemByIndex,
    OMX_MetadataSearchNextItemByKey,
    OMX_MetadataSearchKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
    OMX_MetadataSearchVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_MetadataSearchTypeMax = 0x7fffffff
} OMX_METADATASEARCHMODETYPE;
/** @ingroup metadata */
typedef struct OMX_CONFIG_METADATAITEMCOUNTTYPE
{
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_METADATASCOPETYPE eScopeMode;
    OMX_U32 nScopeSpecifier;
    OMX_U32 nMetadataItemCount;
} OMX_CONFIG_METADATAITEMCOUNTTYPE;

/** @ingroup metadata */
typedef struct OMX_CONFIG_METADATAITEMTYPE
{
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_METADATASCOPETYPE eScopeMode;
    OMX_U32 nScopeSpecifier;
    OMX_U32 nMetadataItemIndex;
    OMX_METADATASEARCHMODETYPE eSearchMode;
    OMX_METADATACHARSETTYPE eKeyCharset;
    OMX_U8 nKeySizeUsed;
    OMX_U8 nKey[128];
    OMX_METADATACHARSETTYPE eValueCharset;
    OMX_STRING sLanguageCountry;
    OMX_U32 nValueMaxSize;
    OMX_U32 nValueSizeUsed;
    OMX_U8 nValue[1];
} OMX_CONFIG_METADATAITEMTYPE;

/* @ingroup metadata */
typedef struct OMX_CONFIG_CONTAINERNODECOUNTTYPE
{
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_BOOL bAllKeys;
    OMX_U32 nParentNodeID;
    OMX_U32 nNumNodes;
} OMX_CONFIG_CONTAINERNODECOUNTTYPE;

/** @ingroup metadata */
typedef struct OMX_CONFIG_CONTAINERNODEIDTYPE
{
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_BOOL bAllKeys;
    OMX_U32 nParentNodeID;
    OMX_U32 nNodeIndex;
    OMX_U32 nNodeID;
    OMX_STRING cNodeName;
    OMX_BOOL bIsLeafType;
} OMX_CONFIG_CONTAINERNODEIDTYPE;

/** @ingroup metadata */
typedef struct OMX_PARAM_METADATAFILTERTYPE
{
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_BOOL bAllKeys;	/* if true then this structure refers to all keys and
                         * the three key fields below are ignored */
    OMX_METADATACHARSETTYPE eKeyCharset;
    OMX_U32 nKeySizeUsed;
    OMX_U8   nKey [128];
    OMX_U32 nLanguageCountrySizeUsed;
    OMX_U8 nLanguageCountry[128];
    OMX_BOOL bEnabled;	/* if true then key is part of filter (e.g.
                         * retained for query later). If false then
                         * key is not part of filter */
} OMX_PARAM_METADATAFILTERTYPE;

/** The OMX_HANDLETYPE structure defines the component handle.  The component
 *  handle is used to access all of the component's public methods and also
 *  contains pointers to the component's private data area.  The component
 *  handle is initialized by the OMX core (with help from the component)
 *  during the process of loading the component.  After the component is
 *  successfully loaded, the application can safely access any of the
 *  component's public functions (although some may return an error because
 *  the state is inappropriate for the access).
 *
 *  @ingroup comp
 */
typedef struct OMX_COMPONENTTYPE
{
    /** The size of this structure, in bytes.  It is the responsibility
        of the allocator of this structure to fill in this value.  Since
        this structure is allocated by the GetHandle function, this
        function will fill in this value. */
    OMX_U32 nSize;

    /** nVersion is the version of the OMX specification that the structure
        is built against.  It is the responsibility of the creator of this
        structure to initialize this value and every user of this structure
        should verify that it knows how to use the exact version of
        this structure found herein. */
    OMX_VERSIONTYPE nVersion;

    /** pComponentPrivate is a pointer to the component private data area.
        This member is allocated and initialized by the component when the
        component is first loaded.  The application should not access this
        data area. */
    OMX_PTR pComponentPrivate;

    /** pApplicationPrivate is a pointer that is a parameter to the
        OMX_GetHandle method, and contains an application private value
        provided by the IL client.  This application private data is
        returned to the IL Client by OMX in all callbacks */
    OMX_PTR pApplicationPrivate;

    /** refer to OMX_GetComponentVersion in OMX_core.h or the OMX IL
        specification for details on the GetComponentVersion method.
     */
    OMX_ERRORTYPE (*GetComponentVersion)(
            OMX_IN  OMX_HANDLETYPE hComponent,
            OMX_OUT OMX_STRING pComponentName,
            OMX_OUT OMX_VERSIONTYPE* pComponentVersion,
            OMX_OUT OMX_VERSIONTYPE* pSpecVersion,
            OMX_OUT OMX_UUIDTYPE* pComponentUUID);

    /** refer to OMX_SendCommand in OMX_core.h or the OMX IL
        specification for details on the SendCommand method.
     */
    OMX_ERRORTYPE (*SendCommand)(
            OMX_IN  OMX_HANDLETYPE hComponent,
            OMX_IN  OMX_COMMANDTYPE Cmd,
            OMX_IN  OMX_U32 nParam1,
            OMX_IN  OMX_PTR pCmdData);

    /** refer to OMX_GetParameter in OMX_core.h or the OMX IL
        specification for details on the GetParameter method.
     */
    OMX_ERRORTYPE (*GetParameter)(
            OMX_IN  OMX_HANDLETYPE hComponent,
            OMX_IN  OMX_INDEXTYPE nParamIndex,
            OMX_INOUT OMX_PTR pComponentParameterStructure);


    /** refer to OMX_SetParameter in OMX_core.h or the OMX IL
        specification for details on the SetParameter method.
     */
    OMX_ERRORTYPE (*SetParameter)(
            OMX_IN  OMX_HANDLETYPE hComponent,
            OMX_IN  OMX_INDEXTYPE nIndex,
            OMX_IN  OMX_PTR pComponentParameterStructure);


    /** refer to OMX_GetConfig in OMX_core.h or the OMX IL
        specification for details on the GetConfig method.
     */
    OMX_ERRORTYPE (*GetConfig)(
            OMX_IN  OMX_HANDLETYPE hComponent,
            OMX_IN  OMX_INDEXTYPE nIndex,
            OMX_INOUT OMX_PTR pComponentConfigStructure);


    /** refer to OMX_SetConfig in OMX_core.h or the OMX IL
        specification for details on the SetConfig method.
     */
    OMX_ERRORTYPE (*SetConfig)(
            OMX_IN  OMX_HANDLETYPE hComponent,
            OMX_IN  OMX_INDEXTYPE nIndex,
            OMX_IN  OMX_PTR pComponentConfigStructure);


    /** refer to OMX_GetExtensionIndex in OMX_core.h or the OMX IL
        specification for details on the GetExtensionIndex method.
     */
    OMX_ERRORTYPE (*GetExtensionIndex)(
            OMX_IN  OMX_HANDLETYPE hComponent,
            OMX_IN  OMX_STRING cParameterName,
            OMX_OUT OMX_INDEXTYPE* pIndexType);


    /** refer to OMX_GetState in OMX_core.h or the OMX IL
        specification for details on the GetState method.
     */
    OMX_ERRORTYPE (*GetState)(
            OMX_IN  OMX_HANDLETYPE hComponent,
            OMX_OUT OMX_STATETYPE* pState);


    /** The ComponentTunnelRequest method will interact with another OMX
        component to determine if tunneling is possible and to setup the
        tunneling.  The return codes for this method can be used to
        determine if tunneling is not possible, or if tunneling is not
        supported.

        Base profile components (i.e. non-interop) do not support this
        method and should return OMX_ErrorNotImplemented

        The interop profile component MUST support tunneling to another
        interop profile component with a compatible port parameters.
        A component may also support proprietary communication.

        If proprietary communication is supported the negotiation of
        proprietary communication is done outside of OMX in a vendor
        specific way. It is only required that the proper result be
        returned and the details of how the setup is done is left
        to the component implementation.

        When this method is invoked when nPort in an output port, the
        component will:
        1.  Populate the pTunnelSetup structure with the output port's
            requirements and constraints for the tunnel.

        When this method is invoked when nPort in an input port, the
        component will:
        1.  Query the necessary parameters from the output port to
            determine if the ports are compatible for tunneling
        2.  If the ports are compatible, the component should store
            the tunnel step provided by the output port
        3.  Determine which port (either input or output) is the buffer
            supplier, and call OMX_SetParameter on the output port to
            indicate this selection.

        The component will return from this call within 5 msec.

        @param [in] hComp
            Handle of the component to be accessed.  This is the component
            handle returned by the call to the OMX_GetHandle method.
        @param [in] nPort
            nPort is used to select the port on the component to be used
            for tunneling.
        @param [in] hTunneledComp
            Handle of the component to tunnel with.  This is the component
            handle returned by the call to the OMX_GetHandle method.  When
            this parameter is 0x0 the component should setup the port for
            communication with the application / IL Client.
        @param [in] nPortOutput
            nPortOutput is used indicate the port the component should
            tunnel with.
        @param [in] pTunnelSetup
            Pointer to the tunnel setup structure.  When nPort is an output port
            the component should populate the fields of this structure.  When
            When nPort is an input port the component should review the setup
            provided by the component with the output port.
        @return OMX_ERRORTYPE
            If the command successfully executes, the return code will be
            OMX_ErrorNone.  Otherwise the appropriate OMX error will be returned.
        @ingroup tun
    */

    OMX_ERRORTYPE (*ComponentTunnelRequest)(
        OMX_IN  OMX_HANDLETYPE hComp,
        OMX_IN  OMX_U32 nPort,
        OMX_IN  OMX_HANDLETYPE hTunneledComp,
        OMX_IN  OMX_U32 nTunneledPort,
        OMX_INOUT  OMX_TUNNELSETUPTYPE* pTunnelSetup);

    /** refer to OMX_UseBuffer in OMX_core.h or the OMX IL
        specification for details on the UseBuffer method.
        @ingroup buf
     */
    OMX_ERRORTYPE (*UseBuffer)(
            OMX_IN OMX_HANDLETYPE hComponent,
            OMX_INOUT OMX_BUFFERHEADERTYPE** ppBufferHdr,
            OMX_IN OMX_U32 nPortIndex,
            OMX_IN OMX_PTR pAppPrivate,
            OMX_IN OMX_U32 nSizeBytes,
            OMX_IN OMX_U8* pBuffer);

    /** refer to OMX_AllocateBuffer in OMX_core.h or the OMX IL
        specification for details on the AllocateBuffer method.
        @ingroup buf
     */
    OMX_ERRORTYPE (*AllocateBuffer)(
            OMX_IN OMX_HANDLETYPE hComponent,
            OMX_INOUT OMX_BUFFERHEADERTYPE** ppBuffer,
            OMX_IN OMX_U32 nPortIndex,
            OMX_IN OMX_PTR pAppPrivate,
            OMX_IN OMX_U32 nSizeBytes);

    /** refer to OMX_FreeBuffer in OMX_core.h or the OMX IL
        specification for details on the FreeBuffer method.
        @ingroup buf
     */
    OMX_ERRORTYPE (*FreeBuffer)(
            OMX_IN  OMX_HANDLETYPE hComponent,
            OMX_IN  OMX_U32 nPortIndex,
            OMX_IN  OMX_BUFFERHEADERTYPE* pBuffer);

    /** refer to OMX_EmptyThisBuffer in OMX_core.h or the OMX IL
        specification for details on the EmptyThisBuffer method.
        @ingroup buf
     */
    OMX_ERRORTYPE (*EmptyThisBuffer)(
            OMX_IN  OMX_HANDLETYPE hComponent,
            OMX_IN  OMX_BUFFERHEADERTYPE* pBuffer);

    /** refer to OMX_FillThisBuffer in OMX_core.h or the OMX IL
        specification for details on the FillThisBuffer method.
        @ingroup buf
     */
    OMX_ERRORTYPE (*FillThisBuffer)(
            OMX_IN  OMX_HANDLETYPE hComponent,
            OMX_IN  OMX_BUFFERHEADERTYPE* pBuffer);

    /** The SetCallbacks method is used by the core to specify the callback
        structure from the application to the component.  This is a blocking
        call.  The component will return from this call within 5 msec.
        @param [in] hComponent
            Handle of the component to be accessed.  This is the component
            handle returned by the call to the GetHandle function.
        @param [in] pCallbacks
            pointer to an OMX_CALLBACKTYPE structure used to provide the
            callback information to the component
        @param [in] pAppData
            pointer to an application defined value.  It is anticipated that
            the application will pass a pointer to a data structure or a "this
            pointer" in this area to allow the callback (in the application)
            to determine the context of the call
        @return OMX_ERRORTYPE
            If the command successfully executes, the return code will be
            OMX_ErrorNone.  Otherwise the appropriate OMX error will be returned.
     */
    OMX_ERRORTYPE (*SetCallbacks)(
            OMX_IN  OMX_HANDLETYPE hComponent,
            OMX_IN  OMX_CALLBACKTYPE* pCallbacks,
            OMX_IN  OMX_PTR pAppData);

    /** ComponentDeInit method is used to deinitialize the component
        providing a means to free any resources allocated at component
        initialization.  NOTE:  After this call the component handle is
        not valid for further use.
        @param [in] hComponent
            Handle of the component to be accessed.  This is the component
            handle returned by the call to the GetHandle function.
        @return OMX_ERRORTYPE
            If the command successfully executes, the return code will be
            OMX_ErrorNone.  Otherwise the appropriate OMX error will be returned.
     */
    OMX_ERRORTYPE (*ComponentDeInit)(
            OMX_IN  OMX_HANDLETYPE hComponent);

    /** @ingroup buf */
    OMX_ERRORTYPE (*UseEGLImage)(
            OMX_IN OMX_HANDLETYPE hComponent,
            OMX_INOUT OMX_BUFFERHEADERTYPE** ppBufferHdr,
            OMX_IN OMX_U32 nPortIndex,
            OMX_IN OMX_PTR pAppPrivate,
            OMX_IN void* eglImage);

    OMX_ERRORTYPE (*ComponentRoleEnum)(
        OMX_IN OMX_HANDLETYPE hComponent,
		OMX_OUT OMX_U8 *cRole,
		OMX_IN OMX_U32 nIndex);

} OMX_COMPONENTTYPE;

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
/* File EOF */
