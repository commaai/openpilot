// =============================================================================
//
// Copyright (c) 2019 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
// =============================================================================

#ifndef PSNPE_HPP
#define PSNPE_HPP

#include <cstdlib>
#include <unordered_map>
#include <functional>
#include "SNPE/SNPE.hpp"
#include "DlSystem/UserBufferMap.hpp"
#include "DlContainer/IDlContainer.hpp"
#include "DlSystem/DlEnums.hpp"
#include "DlSystem/ZdlExportDefine.hpp"

#include "UserBufferList.hpp"
#include "RuntimeConfigList.hpp"
#include "ApplicationBufferMap.hpp"

namespace zdl
{
namespace PSNPE
{

/** @addtogroup c_plus_plus_apis C++
@{ */

/**
 *@ brief build snpe instance in serial or parallel
 *
 */
enum ZDL_EXPORT BuildMode {
   SERIAL = 0,
   PARALLEL = 1
};
/**
 * @brief  Input and output transmission mode
 */
enum ZDL_EXPORT InputOutputTransmissionMode
{
   sync = 0,
   outputAsync = 1,
   inputOutputAsync = 2
};

/**
 * @brief  A structure representing parameters of callback function of Async Output mode
 */
struct ZDL_EXPORT OutputAsyncCallbackParam
{
   size_t dataIndex;
   bool executeStatus;
   OutputAsyncCallbackParam(size_t _index,bool _status)
     : dataIndex(_index),executeStatus(_status){};
};
/**
 * @brief  A structure representing parameters of callback function of Async Input/Output mode
 */
struct ZDL_EXPORT InputOutputAsyncCallbackParam
{
   size_t dataIndex;
   const ApplicationBufferMap& outputMap;
   bool executeStatus;
   InputOutputAsyncCallbackParam(size_t _index, const ApplicationBufferMap& output_map,bool _status)
     : dataIndex(_index)
     , outputMap(output_map)
     ,executeStatus(_status){

     };
};
using OutputAsyncCallbackFunc = std::function<void(OutputAsyncCallbackParam)>;
using InputOutputAsyncCallbackFunc = std::function<void(InputOutputAsyncCallbackParam)>;
/**
 * @brief .
 *
 * A structure BulkSNPE configuration
 */
struct ZDL_EXPORT BuildConfig final
{
   BuildMode buildMode = BuildMode::SERIAL;
   zdl::DlContainer::IDlContainer* container;
   zdl::DlSystem::StringList outputBufferNames;
   RuntimeConfigList runtimeConfigList;
   OutputAsyncCallbackFunc outputCallback;
   InputOutputAsyncCallbackFunc inputOutputCallback;
   InputOutputTransmissionMode inputOutputTransmissionMode = InputOutputTransmissionMode::sync;
};
/**
 * @brief .
 *
 * The class for executing SNPE instances in parallel.
 */
class ZDL_EXPORT PSNPE final
{
 public:
   ~PSNPE();

   explicit PSNPE() noexcept :m_TransmissionMode(InputOutputTransmissionMode::sync){};

   /**
    * @brief Build snpe instances.
    *
    */
   bool build(BuildConfig& buildConfig) noexcept;

   /**
    * @brief Execute snpe instances in Async Output mode and Sync mode
    *
    * @param[in] inputBufferList A list of user buffers that contains the input data
    *
    * @param[in,out] outputBufferList A list of user buffers that will hold the output data
    *
    */
   bool execute(UserBufferList& inputBufferList, UserBufferList& outputBufferList) noexcept;

   /**
    * @brief  Execute snpe instances in Async Input/Output mode
    *
    * @param[in]inputMap A map of input buffers that contains input data. The names of buffers
    *                     need to be matched with names retrived through getInputTensorNames()
    *
    * @param dataIndex Index of the input data
    *
    * @param isTF8buff Whether prefer to using 8 bit quantized element for inference
    *
    * @return True if executed successfully; flase, otherwise.
    */
   bool executeInputOutputAsync(const ApplicationBufferMap& inputMap, size_t dataIndex, bool isTF8buff) noexcept;
   /**
    * @brief Returns the input layer names of the network.
    *
    * @return StringList which contains the input layer names
    */
   const zdl::DlSystem::StringList getInputTensorNames() const noexcept;

   /**
    * @brief Returns the output layer names of the network.
    *
    * @return StringList which contains the output layer names
    */
   const zdl::DlSystem::StringList getOutputTensorNames() const noexcept;

   /**
    * @brief Returns the input tensor dimensions of the network.
    *
    * @return TensorShape which contains the dimensions.
    */
   const zdl::DlSystem::TensorShape getInputDimensions() const noexcept;

   /**
    * @brief Returns attributes of buffers.
    *
    * @see zdl::SNPE
    *
    * @return BufferAttributes of input/output tensor named.
    */
   const zdl::DlSystem::TensorShape getBufferAttributesDims(const char *name) const noexcept;

 private:
   PSNPE(const PSNPE&) = delete;
   PSNPE& operator=(const PSNPE&) = delete;
   zdl::PSNPE::InputOutputTransmissionMode m_TransmissionMode;

};

/** @} */ /* end_addtogroup c_plus_plus_apis C++ */
} // namespace PSNPE
} // namespace zdl
#endif // PSNPE_HPP
