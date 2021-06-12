// =============================================================================
//
// Copyright (c) 2019-2020 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
// =============================================================================

#ifndef PSNPE_HPP
#define PSNPE_HPP

#include <cstdlib>
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
/**
 * @brief  This callback is called when the output data is ready, only use for Output Async mode
 */
using OutputAsyncCallbackFunc = std::function<void(OutputAsyncCallbackParam)>;
/**
 * @brief  This callback is called when the output data is ready, only use for Output-Input Async mode
 */
using InputOutputAsyncCallbackFunc = std::function<void(InputOutputAsyncCallbackParam)>;
/**
 * @brief   This callback is called when the input data is ready,only use for Output-Input Async mode 
 */
using InputOutputAsyncInputCallback = std::function<std::shared_ptr<ApplicationBufferMap>(const std::vector<std::string> &,
    const zdl::DlSystem::StringList &)>;
/**
 * @brief .
 *
 * A structure PSNPE configuration
 *
 */
struct ZDL_EXPORT BuildConfig final
{
   BuildMode buildMode = BuildMode::SERIAL; ///< Specify build in serial mode or parallel mode
   zdl::DlContainer::IDlContainer* container;///< The opened container ptr
   zdl::DlSystem::StringList outputBufferNames;///< Specify the output layer name
   RuntimeConfigList runtimeConfigList;///< The runtime config list for PSNPE, @see RuntimeConfig
   size_t inputThreadNumbers = 1;///< Specify the number of threads used in the execution phase to process input data, only used in inputOutputAsync mode
   size_t outputThreadNumbers = 1;///< Specify the number of threads used in the execution phase to process output data, only used in inputOutputAsync and outputAsync mode
   OutputAsyncCallbackFunc outputCallback;///< The callback to deal with output data ,only used in outputAsync mode
   InputOutputAsyncCallbackFunc inputOutputCallback;///< The callback to deal with output data ,only used in inputOutputAsync mode
   InputOutputAsyncInputCallback inputOutputInputCallback;///< The callback to deal with input data ,only used in inputOutputAsync mode
   InputOutputTransmissionMode inputOutputTransmissionMode = InputOutputTransmissionMode::sync;///< Specify execution mode
   zdl::DlSystem::ProfilingLevel_t profilingLevel = zdl::DlSystem::ProfilingLevel_t::OFF;///< Specify profiling level for Diaglog
   uint64_t encode[2] = {0, 0};
   bool enableInitCache = false;
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
   bool executeInputOutputAsync(const std::vector<std::string>& inputMap, size_t dataIndex, bool isTF8buff) noexcept;
   bool executeInputOutputAsync(const std::vector<std::string>& inputMap, size_t dataIndex, bool isTF8buff,bool isTF8Outputbuff) noexcept;
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

   zdl::DlSystem::Optional<zdl::DlSystem::IBufferAttributes*> getInputOutputBufferAttributes(const char *name) const noexcept;

 private:
   PSNPE(const PSNPE&) = delete;
   PSNPE& operator=(const PSNPE&) = delete;
   zdl::PSNPE::InputOutputTransmissionMode m_TransmissionMode;

};

/** @} */ /* end_addtogroup c_plus_plus_apis C++ */
} // namespace PSNPE
} // namespace zdl
#endif // PSNPE_HPP
