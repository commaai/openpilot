//==============================================================================
//
//  Copyright (c) 2015-2021 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef _SNPE_FACTORY_HPP_
#define _SNPE_FACTORY_HPP_

#include "SNPE/SNPE.hpp"
#include "DlSystem/DlEnums.hpp"
#include "DlSystem/UDLFunc.hpp"
#include "DlSystem/ZdlExportDefine.hpp"
#include "DlSystem/DlOptional.hpp"

namespace zdl {
   namespace DlSystem
   {
      class ITensorFactory;
      class IUserBufferFactory;
   }
   namespace DlContainer
   {
      class IDlContainer;
   }
}



namespace zdl { namespace SNPE {
/** @addtogroup c_plus_plus_apis C++
@{ */

/**
 * The factory class for creating SNPE objects.
 *
 */
class ZDL_EXPORT SNPEFactory
{
public:

   /**
    * Indicates whether the supplied runtime is available on the
    * current platform.
    *
    * @param[in] runtime The target runtime to check.
    *
    * @return True if the supplied runtime is available; false,
    *         otherwise.
    */
   static bool isRuntimeAvailable(zdl::DlSystem::Runtime_t runtime);

   /**
    * Indicates whether the supplied runtime is available on the
    * current platform.
    *
    * @param[in] runtime The target runtime to check.
    *
    * @param[in] option Extent to perform runtime available check.
    *
    * @return True if the supplied runtime is available; false,
    *         otherwise.
    */
   static bool isRuntimeAvailable(zdl::DlSystem::Runtime_t runtime,
                                  zdl::DlSystem::RuntimeCheckOption_t option);

   /**
    * Gets a reference to the tensor factory.
    *
    * @return A reference to the tensor factory.
    */
   static zdl::DlSystem::ITensorFactory& getTensorFactory();

   /**
    * Gets a reference to the UserBuffer factory.
    *
    * @return A reference to the UserBuffer factory.
    */
   static zdl::DlSystem::IUserBufferFactory& getUserBufferFactory();

   /**
    * Gets the version of the SNPE library.
    *
    * @return Version of the SNPE library.
    *
    */
   static zdl::DlSystem::Version_t getLibraryVersion();

   /**
    * Set the SNPE storage location for all SNPE instances in this
    * process. Note that this may only be called once, and if so
    * must be called before creating any SNPE instances.
    *
    * @param[in] storagePath Absolute path to a directory which SNPE may
    *  use for caching and other storage purposes.
    *
    * @return True if the supplied path was succesfully set as
    *  the SNPE storage location, false otherwise.
    */
   static bool setSNPEStorageLocation(const char* storagePath);

   /**
    * @brief Register a user-defined op package with SNPE.
    *
    * @param[in] regLibraryPath Path to the registration library
    *                      that allows clients to register a set of operations that are
    *                      part of the package, and share op info with SNPE
    *
    * @return True if successful, False otherwise.
    */
   static bool addOpPackage( const std::string& regLibraryPath );

   /**
    * Indicates whether the OpenGL and OpenCL interoperability is supported
    * on GPU platform.
    *
    * @return True if the OpenGL and OpenCl interop is supported; false,
    *         otherwise.
    */
   static bool isGLCLInteropSupported();

   static const char* getLastError();

   /**
    * Initializes logging with the specified log level.
    * initializeLogging with level, is used on Android platforms
    * and after successful initialization, SNPE
    * logs are printed in android logcat logs.
    *
    * It is recommended to initializeLogging before creating any
    * SNPE instances, in order to capture information related to
    * core initialization. If this is called again after first
    * time initialization, subsequent calls are ignored.
    * Also, Logging can be re-initialized after a call to
    * terminateLogging API by calling initializeLogging again.
    *
    * A typical usage of Logging life cycle can be
    * initializeLogging()
    *        any other SNPE API like isRuntimeAvailable()
    * * setLogLevel() - optional - can be called anytime
    *         between initializeLogging & terminateLogging
    *		  SNPE instance creation, inference, destroy
    * terminateLogging().
    *
    * Please note, enabling logging can have performance impact.
    *
    * @param[in] LogLevel_t Log level (LOG_INFO, LOG_WARN, etc.).
    *
    * @return True if successful, False otherwise.
    */
   static bool initializeLogging(const zdl::DlSystem::LogLevel_t& level);

   /**
    * Initializes logging with the specified log level and log path.
    * initializeLogging with level & log path, is used on non Android
    * platforms and after successful initialization, SNPE
    * logs are printed in std output & into log files created in the
    * log path.
    *
    * It is recommended to initializeLogging before creating any
    * SNPE instances, in order to capture information related to
    * core initialization. If this is called again after first
    * time initialization, subsequent calls are ignored.
    * Also, Logging can be re-initialized after a call to
    * terminateLogging API by calling initializeLogging again.
    *
    * A typical usage of Logging life cycle can be
    * initializeLogging()
    *        any other SNPE API like isRuntimeAvailable()
    * * setLogLevel() - optional - can be called anytime
    *         between initializeLogging & terminateLogging
    *		  SNPE instance creation, inference, destroy
    * terminateLogging()
    *
    * Please note, enabling logging can have performance impact
    *
    * @param[in] LogLevel_t Log level (LOG_INFO, LOG_WARN, etc.).
    *
    * @param[in] Path of directory to store logs.
    *      If path is empty, the default path is "./Log".
    *      For android, the log path is ignored.
    *
    * @return True if successful, False otherwise.
    */
   static bool initializeLogging(const zdl::DlSystem::LogLevel_t& level, const std::string& logPath);

   /**
    * Updates the current logging level with the specified level.
    * setLogLevel is optional, called anytime after initializeLogging
    * and before terminateLogging, to update the log level set.
    * Log levels can be updated multiple times by calling setLogLevel
    * A call to setLogLevel() is ignored if it is made before
    * initializeLogging() or after terminateLogging()
    *
    * @param[in] LogLevel_t Log level (LOG_INFO, LOG_WARN, etc.).
    *
    * @return True if successful, False otherwise.
    */
   static bool setLogLevel(const zdl::DlSystem::LogLevel_t& level);

   /**
    * Terminates logging.
    *
    * It is recommended to terminateLogging after initializeLogging
    * in order to disable logging information.
    * If this is called before initialization or after first time termination,
    * calls are ignored.
    *
    * @return True if successful, False otherwise.
    */
   static bool terminateLogging(void);
};

/** @} */ /* end_addtogroup c_plus_plus_apis C++ */
}}


#endif
