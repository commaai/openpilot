/*
 * Copyright (C) 2012 The Android Open Source Project
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

#ifndef HDCP_API_H_

#define HDCP_API_H_

#include <utils/Errors.h>
#include <system/window.h>

namespace android {

// Two different kinds of modules are covered under the same HDCPModule
// structure below, a module either implements decryption or encryption.
struct HDCPModule {
    typedef void (*ObserverFunc)(void *cookie, int msg, int ext1, int ext2);

    // The msg argument in calls to the observer notification function.
    enum {
        // Sent in response to a call to "HDCPModule::initAsync" once
        // initialization has either been successfully completed,
        // i.e. the HDCP session is now fully setup (AKE, Locality Check,
        // SKE and any authentication with repeaters completed) or failed.
        // ext1 should be a suitable error code (status_t), ext2 is
        // unused for ENCRYPTION and in the case of HDCP_INITIALIZATION_COMPLETE
        // holds the local TCP port the module is listening on.
        HDCP_INITIALIZATION_COMPLETE,
        HDCP_INITIALIZATION_FAILED,

        // Sent upon completion of a call to "HDCPModule::shutdownAsync".
        // ext1 should be a suitable error code, ext2 is unused.
        HDCP_SHUTDOWN_COMPLETE,
        HDCP_SHUTDOWN_FAILED,

        HDCP_UNAUTHENTICATED_CONNECTION,
        HDCP_UNAUTHORIZED_CONNECTION,
        HDCP_REVOKED_CONNECTION,
        HDCP_TOPOLOGY_EXECEEDED,
        HDCP_UNKNOWN_ERROR,

        // DECRYPTION only: Indicates that a client has successfully connected,
        // a secure session established and the module is ready to accept
        // future calls to "decrypt".
        HDCP_SESSION_ESTABLISHED,
    };

    // HDCPModule capability bit masks
    enum {
        // HDCP_CAPS_ENCRYPT: mandatory, meaning the HDCP module can encrypt
        // from an input byte-array buffer to an output byte-array buffer
        HDCP_CAPS_ENCRYPT = (1 << 0),
        // HDCP_CAPS_ENCRYPT_NATIVE: the HDCP module supports encryption from
        // a native buffer to an output byte-array buffer. The format of the
        // input native buffer is specific to vendor's encoder implementation.
        // It is the same format as that used by the encoder when
        // "storeMetaDataInBuffers" extension is enabled on its output port.
        HDCP_CAPS_ENCRYPT_NATIVE = (1 << 1),
    };

    // Module can call the notification function to signal completion/failure
    // of asynchronous operations (such as initialization) or out of band
    // events.
    HDCPModule(void *cookie, ObserverFunc observerNotify) {};

    virtual ~HDCPModule() {};

    // ENCRYPTION: Request to setup an HDCP session with the host specified
    // by addr and listening on the specified port.
    // DECRYPTION: Request to setup an HDCP session, addr is the interface
    // address the module should bind its socket to. port will be 0.
    // The module will pick the port to listen on itself and report its choice
    // in the "ext2" argument of the HDCP_INITIALIZATION_COMPLETE callback.
    virtual status_t initAsync(const char *addr, unsigned port) = 0;

    // Request to shutdown the active HDCP session.
    virtual status_t shutdownAsync() = 0;

    // Returns the capability bitmask of this HDCP session.
    virtual uint32_t getCaps() {
        return HDCP_CAPS_ENCRYPT;
    }

    // ENCRYPTION only:
    // Encrypt data according to the HDCP spec. "size" bytes of data are
    // available at "inData" (virtual address), "size" may not be a multiple
    // of 128 bits (16 bytes). An equal number of encrypted bytes should be
    // written to the buffer at "outData" (virtual address).
    // This operation is to be synchronous, i.e. this call does not return
    // until outData contains size bytes of encrypted data.
    // streamCTR will be assigned by the caller (to 0 for the first PES stream,
    // 1 for the second and so on)
    // inputCTR _will_be_maintained_by_the_callee_ for each PES stream.
    virtual status_t encrypt(
            const void *inData, size_t size, uint32_t streamCTR,
            uint64_t *outInputCTR, void *outData) {
        return INVALID_OPERATION;
    }

    // Encrypt data according to the HDCP spec. "size" bytes of data starting
    // at location "offset" are available in "buffer" (buffer handle). "size"
    // may not be a multiple of 128 bits (16 bytes). An equal number of
    // encrypted bytes should be written to the buffer at "outData" (virtual
    // address). This operation is to be synchronous, i.e. this call does not
    // return until outData contains size bytes of encrypted data.
    // streamCTR will be assigned by the caller (to 0 for the first PES stream,
    // 1 for the second and so on)
    // inputCTR _will_be_maintained_by_the_callee_ for each PES stream.
    virtual status_t encryptNative(
            buffer_handle_t buffer, size_t offset, size_t size,
            uint32_t streamCTR, uint64_t *outInputCTR, void *outData) {
        return INVALID_OPERATION;
    }
    // DECRYPTION only:
    // Decrypt data according to the HDCP spec.
    // "size" bytes of encrypted data are available at "inData"
    // (virtual address), "size" may not be a multiple of 128 bits (16 bytes).
    // An equal number of decrypted bytes should be written to the buffer
    // at "outData" (virtual address).
    // This operation is to be synchronous, i.e. this call does not return
    // until outData contains size bytes of decrypted data.
    // Both streamCTR and inputCTR will be provided by the caller.
    virtual status_t decrypt(
            const void *inData, size_t size,
            uint32_t streamCTR, uint64_t inputCTR,
            void *outData) {
        return INVALID_OPERATION;
    }

private:
    HDCPModule(const HDCPModule &);
    HDCPModule &operator=(const HDCPModule &);
};

}  // namespace android

// A shared library exporting the following methods should be included to
// support HDCP functionality. The shared library must be called
// "libstagefright_hdcp.so", it will be dynamically loaded into the
// mediaserver process.
extern "C" {
    // Create a module for ENCRYPTION.
    extern android::HDCPModule *createHDCPModule(
            void *cookie, android::HDCPModule::ObserverFunc);

    // Create a module for DECRYPTION.
    extern android::HDCPModule *createHDCPModuleForDecryption(
            void *cookie, android::HDCPModule::ObserverFunc);
}

#endif  // HDCP_API_H_

