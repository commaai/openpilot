//==============================================================================
//
//  Copyright (c) 2015-2017 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef _SNPE_SNPE_HPP_
#define _SNPE_SNPE_HPP_

#include <map>
#include <vector>

#include "DlSystem/DlOptional.hpp"
#include "DlSystem/DlVersion.hpp"
#include "DlSystem/IBufferAttributes.hpp"
#include "DlSystem/ITensor.hpp"
#include "DlSystem/TensorShape.hpp"
#include "DlSystem/TensorMap.hpp"
#include "DlSystem/String.hpp"
#include "DlSystem/StringList.hpp"
#include "DlSystem/IUserBuffer.hpp"
#include "DlSystem/UserBufferMap.hpp"
#include "DlSystem/ZdlExportDefine.hpp"

namespace zdl {
   namespace SNPE
   {
      class SnpeRuntime;
   }
}
namespace zdl {
   namespace DiagLog
   {
      class IDiagLog;
   }
}

namespace zdl { namespace SNPE {
/** @addtogroup c_plus_plus_apis C++
@{ */

/**
 * @brief .
 *
 * The SNPE interface class definition
 */
class ZDL_EXPORT SNPE final
{
public:

   // keep this undocumented to be hidden in doxygen using HIDE_UNDOC_MEMBERS
   explicit SNPE(std::unique_ptr<zdl::SNPE::SnpeRuntime>&& runtime) noexcept;
   ~SNPE();

   /**
    * @brief Gets the names of input tensors to the network
    *
    * To support multiple input scenarios, where multiple tensors are
    * passed through execute() in a TensorMap, each tensor needs to
    * be uniquely named. The names of tensors can be retrieved
    * through this function.
    *
    * In the case of a single input, one name will be returned.
    *
    * @note Note that because the returned value is an Optional list,
    * the list must be verified as boolean true value before being
    * dereferenced.
    *
    * @return An Optional List of input tensor names.
    *
    * @see zdl::DlSystem::Optional
    */
   zdl::DlSystem::Optional<zdl::DlSystem::StringList>
      getInputTensorNames() const noexcept;

    /**
     * @brief Gets the names of output tensors to the network
     *
     * @return List of output tensor names.
     */
   zdl::DlSystem::Optional<zdl::DlSystem::StringList>
      getOutputTensorNames() const noexcept;

   /**
    * @brief Processes the input data and returns the output
    *
    * @param[in] A map of tensors that contains the input data for
    *            each input. The names of tensors needs to be
    *            matched with names retrieved through
    *            getInputTensorNames()
    *
    * @param[in,out] An empty map of tensors that will contain the output
    *                data of potentially multiple layers (the key
    *                in the map is the layer name) upon return
    *
    * @note output tensormap has to be empty.  To forward propagate
    *       and get results in user-supplied tensors, use
    *       executeWithSuppliedOutputTensors.
    */
   bool execute(const zdl::DlSystem::TensorMap &input,
                zdl::DlSystem::TensorMap &output) noexcept;

   /**
    * @brief Processes the input data and returns the output
    *
    * @param[in] A single tensor contains the input data.
    *
    * @param[in,out] An empty map of tensors that will contain the output
    *                data of potentially multiple layers (the key
    *                in the map is the layer name) upon return
    *
    * @note output tensormap has to be empty.
    */
   bool execute(const zdl::DlSystem::ITensor *input,
                zdl::DlSystem::TensorMap &output) noexcept;

   /**
    * @brief Processes the input data and returns the output, using
    *        user-supplied buffers
    *
    * @param[in] A map of UserBuffers that contains the input data for
    *            each input. The names of UserBuffers needs to be
    *            matched with names retrieved through
    *            getInputTensorNames()
    *
    * @param[in,out] A map of UserBuffers that will hold the output
    *                data of potentially multiple layers (the key
    *                in the map is the UserBuffer name)
    *
    * @note input and output UserBuffer maps must be fully pre-populated. with
    *       dimensions matching what the network expects.
    *       For example, if there are 5 output UserBuffers they all have to be
    *       present in map.
    *
    *       Caller must guarantee that for the duration of execute(), the buffer
    *       stored in UserBuffer would remain valid.  For more detail on buffer
    *       ownership and lifetime requirements, please refer to zdl::DlSystem::UserBuffer
    *       documentation.
    */
   bool execute(const zdl::DlSystem::UserBufferMap &input,
                const zdl::DlSystem::UserBufferMap &output) noexcept;

    /**
    * @brief Returns the version string embedded at model conversion
    * time.
    *
    * @return Model version string, which is a free-form string
    *         supplied at the time of the conversion
    *
    */
   zdl::DlSystem::String getModelVersion() const noexcept;

   /**
    * @brief Returns the dimensions of the input data to the model in the
    * form of TensorShape. The dimensions in TensorShape corresponds to
    * what the tensor dimensions would need to be for an input tensor to
    * the model.
    *
    * @param[in] layer input name.
    *
    * @note Note that this function only makes sense for networks 
    *       that have a fixed input size. For networks in which the
    *       input size varies with each call of Execute(), this
    *       function should not be used.
    *
    * @note Because the returned type is an Optional instance, it must
    *       be verified as a boolean true value before being dereferenced.
    * 
    * @return An Optional instance of TensorShape that maintains dimensions,
    *         matching the tensor dimensions for input to the model,
    *         where the last entry is the fastest varying dimension, etc.
    *  
    * @see zdl::DlSystem::ITensor
    * @see zdl::DlSystem::TensorShape
    * @see zdl::DlSystem::Optional
    */
   zdl::DlSystem::Optional<zdl::DlSystem::TensorShape>
      getInputDimensions() const noexcept;
   zdl::DlSystem::Optional<zdl::DlSystem::TensorShape>
      getInputDimensions(const char *name) const noexcept;

   /**
    * @brief Gets the output layer(s) for the network. 
    *  
    * Note that the output layers returned by this function may be 
    * different than those specified when the network was created 
    * via the zdl::SNPE::SNPEBuilder. For example, if the
    * network was created in debug mode with no explicit output 
    * layers specified, this will contain all layers.
    *
    * @note Note that because the returned value is an Optional StringList,
    * the list must be verified as a boolean true value before being
    * dereferenced.
    *
    * @return A List of output layer names.
    *
    * @see zdl::DlSystem::Optional
    */
   zdl::DlSystem::Optional<zdl::DlSystem::StringList>
      getOutputLayerNames() const noexcept;

   /**
     * @brief Returns attributes of buffers used to feed input tensors and receive result from output tensors.
     *
     * @param[in] Tensor name.
     *
     * @return BufferAttributes of input/output tensor named
     */
   zdl::DlSystem::Optional<zdl::DlSystem::IBufferAttributes*> getInputOutputBufferAttributes(const char *name) const noexcept;

   zdl::DlSystem::Optional<zdl::DlSystem::IBufferAttributes*> getInputOutputBufferAttributesTf8(const char *name) const noexcept;

   /**
    * @brief .
    *
    * Get the diagnostic logging interface
    *
    * @note Note that because the returned type is an Optional instance,
    * it must be verified as a boolean true value before being
    * dereferenced.
    *
    * @see zdl::DlSystem::Optional
    */
   zdl::DlSystem::Optional<zdl::DiagLog::IDiagLog*>
      getDiagLogInterface() noexcept;

private:
   SNPE(const SNPE&) = delete;
   SNPE& operator=(const SNPE&) = delete;

   std::unique_ptr<SnpeRuntime> m_Runtime;
};

/** @} */ /* end_addtogroup c_plus_plus_apis C++ */
}}

#endif
