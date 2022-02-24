//=============================================================================
//
//  Copyright (c) 2017 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#ifndef _IUSERBUFFER_FACTORY_HPP
#define _IUSERBUFFER_FACTORY_HPP

#include "IUserBuffer.hpp"
#include "TensorShape.hpp"
#include "ZdlExportDefine.hpp"
#include "DlEnums.hpp"
namespace zdl {
    namespace DlSystem {
        class IUserBuffer;

        class TensorShape;
    }
}

namespace zdl {
namespace DlSystem {

/** @addtogroup c_plus_plus_apis C++
@{ */

/**
* Factory interface class to create IUserBuffer objects.
*/
class ZDL_EXPORT IUserBufferFactory {
public:
    virtual ~IUserBufferFactory() = default;

    /**
     * @brief Creates a UserBuffer
     *
     * @param[in] buffer Pointer to the buffer that the caller supplies
     *
     * @param[in] bufSize Buffer size, in bytes
     *
     * @param[in] strides Total number of bytes between elements in each dimension.
     *          E.g. A tightly packed tensor of floats with dimensions [4, 3, 2] would have strides of [24, 8, 4].
     *
     * @param[in] userBufferEncoding Reference to an UserBufferEncoding object
     *
     * @note Caller has to ensure that memory pointed to by buffer stays accessible
     *       for the lifetime of the object created
     */
    virtual std::unique_ptr<IUserBuffer>
    createUserBuffer(void *buffer, size_t bufSize, const zdl::DlSystem::TensorShape &strides, zdl::DlSystem::UserBufferEncoding* userBufferEncoding) noexcept = 0;

    /**
     * @brief Creates a UserBuffer
     *
     * @param[in] buffer Pointer to the buffer that the caller supplies
     *
     * @param[in] bufSize Buffer size, in bytes
     *
     * @param[in] strides Total number of bytes between elements in each dimension.
     *          E.g. A tightly packed tensor of floats with dimensions [4, 3, 2] would have strides of [24, 8, 4].
     *
     * @param[in] userBufferEncoding Reference to an UserBufferEncoding object
     *
     * @param[in] userBufferSource Reference to an UserBufferSource object
     *
     * @note Caller has to ensure that memory pointed to by buffer stays accessible
     *       for the lifetime of the object created
     */
    virtual std::unique_ptr<IUserBuffer>
    createUserBuffer(void *buffer, size_t bufSize, const zdl::DlSystem::TensorShape &strides, zdl::DlSystem::UserBufferEncoding* userBufferEncoding, zdl::DlSystem::UserBufferSource* userBufferSource) noexcept = 0;
};
/** @} */ /* end_addtogroup c_plus_plus_apis C++ */

}
}


#endif
