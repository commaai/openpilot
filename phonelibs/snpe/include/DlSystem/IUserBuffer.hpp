//==============================================================================
//
// Copyright (c) 2017-2020 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef _IUSER_BUFFER_HPP
#define _IUSER_BUFFER_HPP

#include "TensorShape.hpp"
#include "ZdlExportDefine.hpp"
#include <math.h>

namespace zdl {
namespace DlSystem {

/** @addtogroup c_plus_plus_apis C++
@{ */


/**
  * @brief .
  *
  * A base class buffer encoding type
  */
class ZDL_EXPORT UserBufferEncoding {
public:

    /**
      * @brief .
      *
      * An enum class of all supported element types in a IUserBuffer
      */
    enum class ElementType_t
    {
        /// Unknown element type.
        UNKNOWN         = 0,

        /// Each element is presented by float.
        FLOAT           = 1,

        /// Each element is presented by an unsigned int.
        UNSIGNED8BIT    = 2,

        /// Each element is presented by an 8-bit quantized value.
        TF8             = 10,

        /// Each element is presented by an 16-bit quantized value.
        TF16            = 11
    };

    /**
      * @brief Retrieves the size of the element, in bytes.
      *
      * @return Size of the element, in bytes.
     */
    virtual size_t getElementSize() const noexcept = 0;

    /**
      * @brief Retrieves the element type
      *
      * @return Element type
     */
    ElementType_t getElementType() const noexcept {return m_ElementType;};

    virtual ~UserBufferEncoding() {}

protected:
    UserBufferEncoding(ElementType_t  elementType) : m_ElementType(elementType) {};
private:
    const ElementType_t  m_ElementType;
};

/**
  * @brief .
  *
  * A base class buffer source type
  *
  * @note User buffer from CPU support all kinds of runtimes;
  *       User buffer from GLBUFFER support only GPU runtime.
  */
class ZDL_EXPORT UserBufferSource {
public:
   enum class SourceType_t
   {
      /// Unknown buffer source type.
      UNKNOWN = 0,

      /// The network inputs are from CPU buffer.
      CPU = 1,

      /// The network inputs are from OpenGL buffer.
      GLBUFFER = 2
   };

   /**
     * @brief Retrieves the source type
     *
     * @return Source type
    */
   SourceType_t getSourceType() const noexcept {return m_SourceType;};

protected:
   UserBufferSource(SourceType_t sourceType): m_SourceType(sourceType) {};
private:
   const SourceType_t m_SourceType;
};

/**
  * @brief .
  *
  * An source type where input data is delivered from OpenGL buffer
  */
class ZDL_EXPORT UserBufferSourceGLBuffer : public UserBufferSource{
public:
   UserBufferSourceGLBuffer() : UserBufferSource(SourceType_t::GLBUFFER) {};
};

/**
  * @brief .
  *
  * An encoding type where each element is represented by an unsigned int
  */
class ZDL_EXPORT UserBufferEncodingUnsigned8Bit : public UserBufferEncoding {
public:
    UserBufferEncodingUnsigned8Bit() : UserBufferEncoding(ElementType_t::UNSIGNED8BIT) {};
    size_t getElementSize() const noexcept override;

protected:
    UserBufferEncodingUnsigned8Bit(ElementType_t  elementType) : UserBufferEncoding(elementType) {};

};

/**
  * @brief .
  *
  * An encoding type where each element is represented by a float
  */
class ZDL_EXPORT UserBufferEncodingFloat : public UserBufferEncoding {
public:
    UserBufferEncodingFloat() : UserBufferEncoding(ElementType_t::FLOAT) {};
    size_t getElementSize() const noexcept override;

};

/**
  * @brief .
  *
  * An encoding type where each element is represented by tf8, which is an
  * 8-bit quantizd value, which has an exact representation of 0.0
  */

class ZDL_EXPORT UserBufferEncodingTf8 : public UserBufferEncodingUnsigned8Bit {
public:
    UserBufferEncodingTf8() = delete;
    UserBufferEncodingTf8(unsigned char stepFor0, float stepSize) :
            UserBufferEncodingUnsigned8Bit(ElementType_t::TF8),
            m_StepExactly0(stepFor0),
            m_QuantizedStepSize(stepSize) {};

    UserBufferEncodingTf8(const zdl::DlSystem::UserBufferEncoding &ubEncoding) : UserBufferEncodingUnsigned8Bit(ubEncoding.getElementType()){
            const zdl::DlSystem::UserBufferEncodingTf8* ubEncodingTf8
                            = dynamic_cast <const zdl::DlSystem::UserBufferEncodingTf8*> (&ubEncoding);
            if (ubEncodingTf8) {
                m_StepExactly0 = ubEncodingTf8->getStepExactly0();
                m_QuantizedStepSize = ubEncodingTf8->getQuantizedStepSize();
            }
    }

/**
      * @brief Sets the step value that represents 0
      *
      * @param[in] stepExactly0 The step value that represents 0
      *
     */

    void setStepExactly0(const unsigned char stepExactly0) {
        m_StepExactly0 = stepExactly0;
    }


/**
      * @brief Sets the float value that each step represents
      *
      * @param[in] quantizedStepSize The float value of each step size
      *
     */

    void setQuantizedStepSize(const float quantizedStepSize) {
        m_QuantizedStepSize = quantizedStepSize;
    }


/**
      * @brief Retrieves the step that represents 0.0
      *
      * @return Step value
     */

    unsigned char getStepExactly0() const {
        return m_StepExactly0;
    }


/**
     * Calculates the minimum floating point value that
     * can be represented with this encoding.
     *
     * @return Minimum representable floating point value
     */

    float getMin() const {
        return m_QuantizedStepSize * (0 - m_StepExactly0);
    }


/**
     * Calculates the maximum floating point value that
     * can be represented with this encoding.
     *
     * @return Maximum representable floating point value
     */

    float getMax() const {
        return m_QuantizedStepSize * (255 - m_StepExactly0);
    }


/**
      * @brief Retrieves the step size
      *
      * @return Step size
     */

    float getQuantizedStepSize() const {
        return m_QuantizedStepSize;
    }

private:
    unsigned char m_StepExactly0;

    float m_QuantizedStepSize;
};



class ZDL_EXPORT UserBufferEncodingTfN : public UserBufferEncoding {
public:
   UserBufferEncodingTfN() = delete;
   UserBufferEncodingTfN(uint64_t stepFor0, float stepSize, uint8_t bWidth=8):
                                           UserBufferEncoding(getTypeFromWidth(bWidth)),
                                           bitWidth(bWidth),
                                           m_StepExactly0(stepFor0),
                                           m_QuantizedStepSize(stepSize){};

   UserBufferEncodingTfN(const zdl::DlSystem::UserBufferEncoding &ubEncoding) : UserBufferEncoding(ubEncoding.getElementType()){
            const zdl::DlSystem::UserBufferEncodingTfN* ubEncodingTfN
                            = dynamic_cast <const zdl::DlSystem::UserBufferEncodingTfN*> (&ubEncoding);
            if (ubEncodingTfN) {
                m_StepExactly0 = ubEncodingTfN->getStepExactly0();
                m_QuantizedStepSize = ubEncodingTfN->getQuantizedStepSize();
                bitWidth = ubEncodingTfN->bitWidth;
            }
   }

   size_t getElementSize() const noexcept override;
   /**
      * @brief Sets the step value that represents 0
      *
      * @param[in] stepExactly0 The step value that represents 0
      *
     */
   void setStepExactly0(uint64_t stepExactly0) {
      m_StepExactly0 = stepExactly0;
   }

   /**
     * @brief Sets the float value that each step represents
     *
     * @param[in] quantizedStepSize The float value of each step size
     *
    */
   void setQuantizedStepSize(const float quantizedStepSize) {
      m_QuantizedStepSize = quantizedStepSize;
   }

   /**
     * @brief Retrieves the step that represents 0.0
     *
     * @return Step value
    */
   uint64_t getStepExactly0() const {
      return m_StepExactly0;
   }

   /**
    * Calculates the minimum floating point value that
    * can be represented with this encoding.
    *
    * @return Minimum representable floating point value
    */
   float getMin() const {
      return static_cast<float>(m_QuantizedStepSize * (0 - (double)m_StepExactly0));
   }

   /**
    * Calculates the maximum floating point value that
    * can be represented with this encoding.
    *
    * @return Maximum representable floating point value
    */
   float getMax() const{
       return static_cast<float>(m_QuantizedStepSize * (pow(2,bitWidth)-1 - (double)m_StepExactly0));
   };

   /**
     * @brief Retrieves the step size
     *
     * @return Step size
    */
   float getQuantizedStepSize() const {
      return m_QuantizedStepSize;
   }

   ElementType_t getTypeFromWidth(uint8_t width);

   uint8_t bitWidth;
private:
   uint64_t m_StepExactly0;
   float m_QuantizedStepSize;
};


/**
 * @brief UserBuffer contains a pointer and info on how to walk it and interpret its content.
 */
class ZDL_EXPORT IUserBuffer {
public:
    virtual ~IUserBuffer() = default;
    
    /**
      * @brief Retrieves the total number of bytes between elements in each dimension if
      * the buffer were to be interpreted as a multi-dimensional array.
      *
      * @return Number of bytes between elements in each dimension.
      * e.g. A tightly packed tensor of floats with dimensions [4, 3, 2] would
      * return strides of [24, 8, 4].
     */
    virtual const TensorShape& getStrides() const = 0;

    /**
      * @brief Retrieves the size of the buffer, in bytes.
      *
      * @return Size of the underlying buffer, in bytes.
     */
    virtual size_t getSize() const = 0;

    /**
      * @brief Retrieves the size of the inference data in the buffer, in bytes.
      *
      * The inference results from a dynamic-sized model may not be exactly the same size
      * as the UserBuffer provided to SNPE. This function can be used to get the amount
      * of output inference data, which may be less or greater than the size of the UserBuffer.
      *
      * If the inference results fit in the UserBuffer, getOutputSize() would be less than
      * or equal to getSize(). But if the inference results were more than the capacity of
      * the provided UserBuffer, the results would be truncated to fit the UserBuffer. But,
      * getOutputSize() would be greater than getSize(), which indicates a bigger buffer
      * needs to be provided to SNPE to hold all of the inference results.
      *
      * @return Size required for the buffer to hold all inference results, which can be less
      * or more than the size of the buffer, in bytes.
    */
    virtual size_t getOutputSize() const = 0;

    /**
      * @brief Changes the underlying memory that backs the UserBuffer.
      *
      * This can be used to avoid creating multiple UserBuffer objects
      * when the only thing that differs is the memory location.
      *
      * @param[in] buffer Pointer to the memory location
      *
      * @return Whether the set succeeds.
     */
    virtual bool setBufferAddress(void *buffer) noexcept = 0;

    /**
      * @brief Gets a const reference to the data encoding object of
      *        the underlying buffer
      *
      * This is necessary when the UserBuffer is filled by SNPE with
      * data types such as TF8, where the caller needs to know the quantization
      * parameters in order to interpret the data properly
      *
      * @return A read-only encoding object
     */
    virtual const UserBufferEncoding& getEncoding() const noexcept = 0;

    /**
      * @brief Gets a reference to the data encoding object of
      *        the underlying buffer
      *
      * This is necessary when the UserBuffer is re-used, and the encoding
      * parameters can change.  For example, each input can be quantized with
      * different step sizes.
      *
      * @return Data encoding meta-data
     */
    virtual UserBufferEncoding& getEncoding() noexcept = 0;

};

/** @} */ /* end_addtogroup c_plus_plus_apis C++ */

}
}

#endif
