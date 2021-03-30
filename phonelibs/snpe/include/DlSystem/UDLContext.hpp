//==============================================================================
//
// Copyright (c) 2016 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef UDL_CONTEXT_HPP
#define UDL_CONTEXT_HPP

#include <cstring> // memset
#include <tuple>

#include "ZdlExportDefine.hpp"

namespace zdl { namespace DlSystem {
/** @addtogroup c_plus_plus_apis C++
@{ */

/**
 * @brief .
 *
 * UDLContext holds the user defined layer context which 
 * consists of a layer name, layer ID, blob and blob size. 
 *  
 * An instance of UDLContext is passed as an argument to the
 * UDLFactoryFunc provided by the user every time the SNPE 
 * runtime encounters an unknown layer descriptor. The instance 
 * of a UDLContext is created by the SNPE runtime and is 
 * consumed by the user's factory function. The user should 
 * obtain a copy of this class and should not assume any 
 * prolonged object lifetime beyond the UDLFactoryFunction.
 */
class ZDL_EXPORT UDLContext final {
public:
   /**
    * @brief Constructor
    *
    * @param[in] name name of the layer
    *
    * @param[in] type layer type
    *
    * @param[in] id identifier for the layer
    *
    * @param[in] id Blob/bytes as packed by the user code as part of
    *           the Python converter script
    */
   UDLContext(const std::string& name,
              const std::string& type,
              int32_t id,
              const std::string& blob) :
      m_Name(name), m_Type(type), m_Size(blob.size()), m_Id(id) {
      // FIXME not dealing with alloc error
      m_Buffer = new uint8_t[m_Size];
      std::memcpy(m_Buffer, blob.data(), m_Size);
   }

   /**
    * @brief .
    *
    * Empty constructor is useful for
    * creating an empty UDLContext and then run copy constructor
    * from a fully initialized one.
    */
   explicit UDLContext() {}

   /**
    * @brief .
    *
    * destructor Deallocates any internal allocated memory
    */
   ~UDLContext() { release(); }

   /**
    * @brief .
    *
    * Deallocate any internally allocated memory
    */
   void release() {
      if (m_Buffer && m_Size)
         std::memset(m_Buffer, 0, m_Size);
      delete []m_Buffer;
      m_Buffer = nullptr;
      m_Size = 0;
   }

   /**
    * @brief .
    *
    * Copy Constructor - makes a copy from ctx
    *
    * @param[in] ctx Source UDLContext to copy from
    */
   UDLContext(const UDLContext& ctx) : m_Name(ctx.m_Name),
      m_Type(ctx.m_Type),
      m_Id(ctx.m_Id) {
      std::tuple<uint8_t*, size_t> cpy = ctx.getCopy();
      // current compiler does not support get<type>
      m_Buffer = std::get<0>(cpy);
      m_Size = std::get<1>(cpy);
   }

   /**
    * @brief 
    *
    * Assignment operator - makes a copy from ctx
    *
    * @param[in] ctx Source UDLContext to copy from
    *
    * @return this
    */
   UDLContext& operator=(const UDLContext& ctx) {
      UDLContext c (ctx);
      this->swap(c); // non throwing swap
      return *this;
   }

   /**
    * @brief .
    *
    * Move Constructor - Move internals from ctx into this
    *
    * @param[in] ctx Source UDLContext to move from
    */
   UDLContext(UDLContext&& ctx) :
      m_Name(std::move(ctx.m_Name)),
      m_Type(std::move(ctx.m_Type)),
      m_Buffer(ctx.m_Buffer),
      m_Size(ctx.m_Size),
      m_Id(ctx.m_Id) {
      ctx.clear();
   }

   /**
    * @brief .
    *
    * Assignment move - Move assignment operator from ctx
    *
    * @param[in] ctx Source UDLContext to move from
    *
    * @return this
    */
   UDLContext& operator=(UDLContext&& ctx) {
      m_Name = std::move(ctx.m_Name);
      m_Type = std::move(ctx.m_Type);
      m_Buffer = ctx.m_Buffer;
      m_Size = ctx.m_Size;
      m_Id = ctx.m_Id;
      ctx.clear();
      return *this;
   }

   /**
    * @brief .
    *
    * Obtain the name of the layer
    *
    * @return const reference to the name of the layer
    */
   const std::string& getName() const noexcept { return m_Name; }

   /**
    * @brief .
    *
    * Obtain the type of the layer
    *
    * @return const reference to the type of the layer
    */
   const std::string& getType() const noexcept { return m_Type; }

   /**
    * @brief .
    *
    * Obtain the Id of the layer
    *
    * @return The id of the layer
    */
   int32_t getId() const noexcept  { return m_Id; }

   /**
    * @brief .
    *
    * Obtain the size of the blob
    *
    * @return Size of the internal blob
    */
   size_t getSize() const noexcept { return m_Size; }

   /**
    * @brief .
    *
    * Get a const pointer to the internal blob
    *
    * @return Const pointer to the internal blob
    */
   const uint8_t* getBlob() const noexcept { return m_Buffer; }

   /**
    * @brief .
    *
    * Get a copy of the blob/size into a tuple
    *
    * @return A tuple with a pointer to a copy of the blob and a
    *         size
    */
   std::tuple<uint8_t*, size_t> getCopy() const {
      uint8_t* buf = new uint8_t[m_Size];
      // FIXME missing memcpy
      std::memcpy(buf, m_Buffer, m_Size);
      return std::make_tuple(buf, m_Size);
   }

   /**
    * @brief .
    *
    * Set zeros in the internals members
    */
   void clear() {
      m_Name.clear();
      m_Type.clear();
      m_Buffer = 0;
      m_Size = 0;
      m_Id = -1;
   }
private:
   void swap(UDLContext& c) noexcept {
      std::swap(m_Name, c.m_Name);
      std::swap(m_Type, c.m_Type);
      std::swap(m_Id,   c.m_Id);
      std::swap(m_Buffer, c.m_Buffer);
      std::swap(m_Size, c.m_Size);
   }
   std::string m_Name; // name of the layer instance
   std::string m_Type; // The actual layer type
   uint8_t*    m_Buffer = nullptr;
   size_t      m_Size = 0;
   int32_t     m_Id = -1;
};
/** @} */ /* end_addtogroup c_plus_plus_apis C++ */

}}

#endif /* UDL_CONTEXT_HPP */
