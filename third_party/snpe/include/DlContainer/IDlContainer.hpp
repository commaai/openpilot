//=============================================================================
//
//  Copyright (c) 2015-2020 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#ifndef ZEROTH_IDNC_CONTAINER_HPP
#define ZEROTH_IDNC_CONTAINER_HPP

#include <memory>
#include <stdint.h>
#include <string>
#include <vector>
#include <set>

#include "DlSystem/ZdlExportDefine.hpp"
#include "DlSystem/String.hpp"

namespace zdl {
namespace DlContainer {

/** @addtogroup c_plus_plus_apis C++
@{ */

class IDlContainer;
class dlc_error;

/**
 * The structure of a record in a DL container.
 */
struct ZDL_EXPORT DlcRecord
{
   /// Name of the record.
   std::string name;
   /// Byte blob holding the data for the record.
   std::vector<uint8_t> data;

   DlcRecord();
   DlcRecord( DlcRecord&& other )
      : name(std::move(other.name))
      , data(std::move(other.data))
   {}
   DlcRecord(const std::string& new_name)
      : name(new_name)
      , data()
   {
      if(name.empty())
      {
         name.reserve(1);
      }
   }
   DlcRecord(const DlcRecord&) = delete;
};

// The maximum length of any record name.
extern const uint32_t RECORD_NAME_MAX_SIZE;
// The maximum size of the record payload (bytes).
extern const uint32_t RECORD_DATA_MAX_SIZE;
// The maximum number of records in an archive at one time.
extern const uint32_t ARCHIVE_MAX_RECORDS;

/**
 * Represents a container for a neural network model which can
 * be used to load the model into the SNPE runtime.
 */
class ZDL_EXPORT IDlContainer
{
public:
   /**
    * Initializes a container from a container archive file.
    *
    * @param[in] filename Container archive file path.
    *
    * @return A pointer to the initialized container
    */
   static std::unique_ptr<IDlContainer>
   open(const std::string &filename) noexcept;

   /**
    * Initializes a container from a container archive file.
    *
    * @param[in] filename Container archive file path.
    *
    * @return A pointer to the initialized container
    */
   static std::unique_ptr<IDlContainer>
   open(const zdl::DlSystem::String &filename) noexcept;

   /**
    * Initializes a container from a byte buffer.
    *
    * @param[in] buffer Byte buffer holding the contents of an archive
    *                   file.
    *
    * @return A pointer to the initialized container
    */
   static std::unique_ptr<IDlContainer>
   open(const std::vector<uint8_t> &buffer) noexcept;

   /**
    * Initializes a container from a byte buffer.
    *
    * @param[in] buffer Byte buffer holding the contents of an archive
    *                   file.
    *
    * @param[in] size Size of the byte buffer.
    *
    * @return A pointer to the initialized container
    */
   static std::unique_ptr<IDlContainer>
   open(const uint8_t* buffer, const size_t size) noexcept;


/** @} */ /* end_addtogroup c_plus_plus_apis C++ */

   /**
    * Get the record catalog for a container.
    *
    * @param[out] catalog Buffer that will hold the record names on
    *                    return.
    */
   virtual void getCatalog(std::set<std::string> &catalog) const = 0;

    /**
     * Get the record catalog for a container.
     *
     * @param[out] catalog Buffer that will hold the record names on
     *                    return.
     */
   virtual void getCatalog(std::set<zdl::DlSystem::String> &catalog) const = 0;

   /**
    * Get a record from a container by name.
    *
    * @param[in] name Name of the record to fetch.
    * @param[out] record The passed in record will be populated with the
    *                   record data on return. Note that the caller
    *                   will own the data in the record and is
    *                   responsible for freeing it if needed.
    */
   virtual void getRecord(const std::string &name, DlcRecord &record) const = 0;

   /**
    * Get a record from a container by name.
    *
    * @param[in] name Name of the record to fetch.
    * @param[out] record The passed in record will be populated with the
    *                   record data on return. Note that the caller
    *                   will own the data in the record and is
    *                   responsible for freeing it if needed.
    */
   virtual void getRecord(const zdl::DlSystem::String &name, DlcRecord &record) const = 0;

   /**
    * Save the container to an archive on disk. This function will save the
    * container if the filename is different from the file that it was opened
    * from, or if at least one record was modified since the container was
    * opened.
    *
    * It will truncate any existing file at the target path.
    *
    * @param filename Container archive file path.
    *
    * @return indication of success/failure
    */
   virtual bool save(const std::string &filename) = 0;

   /**
    * Save the container to an archive on disk. This function will save the
    * container if the filename is different from the file that it was opened
    * from, or if at least one record was modified since the container was
    * opened.
    *
    * It will truncate any existing file at the target path.
    *
    * @param filename Container archive file path.
    *
    * @return indication of success/failure
    */
   virtual bool save (const zdl::DlSystem::String &filename) = 0;

   virtual ~IDlContainer() {}
};

} // ns DlContainer
} // ns zdl


#endif
