//=============================================================================
//
//  Copyright (c) 2017-2018 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#ifndef _DL_SYSTEM_PLATFORM_CONFIG_HPP_
#define _DL_SYSTEM_PLATFORM_CONFIG_HPP_

#include "DlSystem/ZdlExportDefine.hpp"
#include <string>

namespace zdl{
namespace DlSystem
{

/** @addtogroup c_plus_plus_apis C++
@{ */


/**
  * @brief .
  *
  * A structure OpenGL configuration
  *
  * @note When certain OpenGL context and display are provided to UserGLConfig for using
  *       GPU buffer as input directly, the user MUST ensure the particular OpenGL
  *       context and display remain vaild throughout the execution of neural network models.
  */
struct ZDL_EXPORT UserGLConfig
{
   /// Holds user EGL context.
   ///
   void* userGLContext = nullptr;

   /// Holds user EGL display.
   void* userGLDisplay = nullptr;
};

/**
  * @brief .
  *
  * A structure Gpu configuration
  */
struct ZDL_EXPORT UserGpuConfig{
   /// Holds user OpenGL configuration.
   ///
   UserGLConfig userGLConfig;
};

/**
  * @brief .
  *
  * A class user platform configuration
  */
class ZDL_EXPORT PlatformConfig
{
public:

   /**
     * @brief .
     *
     * An enum class of all supported platform types
     */
   enum class PlatformType_t
   {
      /// Unknown platform type.
      UNKNOWN = 0,

      /// Snapdragon CPU.
      CPU = 1,

      /// Adreno GPU.
      GPU = 2,

      /// Hexagon DSP.
      DSP = 3
   };

   /**
     * @brief .
     *
     * A union class user platform configuration information
     */
   union PlatformConfigInfo
   {
      /// Holds user GPU Configuration.
      ///
      UserGpuConfig userGpuConfig;

      PlatformConfigInfo(){};
   };

   PlatformConfig() : m_PlatformType(PlatformType_t::UNKNOWN),
                      m_PlatformOptions("") {};

   /**
     * @brief Retrieves the platform type
     *
     * @return Platform type
     */
   PlatformType_t getPlatformType() const {return m_PlatformType;};

   /**
     * @brief Indicates whther the plaform configuration is valid.
     *
     * @return True if the platform configuration is valid; false otherwise.
     */
   bool isValid() const {return (PlatformType_t::UNKNOWN != m_PlatformType);};

   /**
     * @brief Retrieves the Gpu configuration
     *
     * @param[out] userGpuConfig The passed in userGpuConfig populated with the Gpu configuration on return.
     *
     * @return True if Gpu configuration was retrieved; false otherwise.
     */
   bool getUserGpuConfig(UserGpuConfig& userGpuConfig) const
   {
      if(m_PlatformType == PlatformType_t::GPU)
      {
         userGpuConfig = m_PlatformConfigInfo.userGpuConfig;
         return true;
      }
      else
      {
         return false;
      }
   }

   /**
     * @brief Sets the Gpu configuration
     *
     * @param[in] userGpuConfig Gpu Configuration
     *
     * @return True if Gpu configuration was successfully set; false otherwise.
     */
   bool setUserGpuConfig(UserGpuConfig& userGpuConfig)
   {
      if((userGpuConfig.userGLConfig.userGLContext != nullptr) && (userGpuConfig.userGLConfig.userGLDisplay != nullptr))
      {
         switch (m_PlatformType)
         {
         case PlatformType_t::GPU:
            m_PlatformConfigInfo.userGpuConfig = userGpuConfig;
            return true;
         case PlatformType_t::UNKNOWN:
            m_PlatformType = PlatformType_t::GPU;
            m_PlatformConfigInfo.userGpuConfig = userGpuConfig;
            return true;
         default:
            return false;
         }
      }
      else
         return false;
   }

   /**
     * @brief Sets the platform options
     *
     * @param[in] options Options as a string in the form of "keyword:options"
     *
     * @return True if options are pass validation; otherwise false.  If false, the options are not updated.
     */
   bool setPlatformOptions(std::string options) {
      std::string oldOptions = m_PlatformOptions;
      m_PlatformOptions = options;
      if (isOptionsValid()) {
         return true;
      } else {
         m_PlatformOptions = oldOptions;
         return false;
      }
   }

   /**
     * @brief Indicates whther the plaform configuration is valid.
     *
     * @return True if the platform configuration is valid; false otherwise.
     */
   bool isOptionsValid() const;

   /**
     * @brief Gets the platform options
     *
     * @return Options as a string
     */
   std::string getPlatformOptions() const { return m_PlatformOptions; }

private:
   PlatformType_t m_PlatformType;
   PlatformConfigInfo m_PlatformConfigInfo;
   std::string m_PlatformOptions;
};

/** @} */ /* end_addtogroup c_plus_plus_apis C++ */

}} //namespace end

#endif
