// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2006-2009 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2010 Hauke Heibel <hauke.heibel@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_MATRIXSTORAGE_H
#define EIGEN_MATRIXSTORAGE_H

#ifdef EIGEN_DENSE_STORAGE_CTOR_PLUGIN
  #define EIGEN_INTERNAL_DENSE_STORAGE_CTOR_PLUGIN EIGEN_DENSE_STORAGE_CTOR_PLUGIN;
#else
  #define EIGEN_INTERNAL_DENSE_STORAGE_CTOR_PLUGIN
#endif

namespace Eigen {

namespace internal {

struct constructor_without_unaligned_array_assert {};

/** \internal
  * Static array. If the MatrixOrArrayOptions require auto-alignment, the array will be automatically aligned:
  * to 16 bytes boundary if the total size is a multiple of 16 bytes.
  */
template <typename T, int Size, int MatrixOrArrayOptions,
          int Alignment = (MatrixOrArrayOptions&DontAlign) ? 0
                        : (((Size*sizeof(T))%16)==0) ? 16
                        : 0 >
struct plain_array
{
  T array[Size];

  plain_array() 
  { 
    EIGEN_STATIC_ASSERT(Size * sizeof(T) <= 128 * 128 * 8, OBJECT_ALLOCATED_ON_STACK_IS_TOO_BIG);
  }

  plain_array(constructor_without_unaligned_array_assert) 
  { 
    EIGEN_STATIC_ASSERT(Size * sizeof(T) <= 128 * 128 * 8, OBJECT_ALLOCATED_ON_STACK_IS_TOO_BIG);
  }
};

#if defined(EIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT)
  #define EIGEN_MAKE_UNALIGNED_ARRAY_ASSERT(sizemask)
#elif EIGEN_GNUC_AT_LEAST(4,7) 
  // GCC 4.7 is too aggressive in its optimizations and remove the alignement test based on the fact the array is declared to be aligned.
  // See this bug report: http://gcc.gnu.org/bugzilla/show_bug.cgi?id=53900
  // Hiding the origin of the array pointer behind a function argument seems to do the trick even if the function is inlined:
  template<typename PtrType>
  EIGEN_ALWAYS_INLINE PtrType eigen_unaligned_array_assert_workaround_gcc47(PtrType array) { return array; }
  #define EIGEN_MAKE_UNALIGNED_ARRAY_ASSERT(sizemask) \
    eigen_assert((reinterpret_cast<size_t>(eigen_unaligned_array_assert_workaround_gcc47(array)) & sizemask) == 0 \
              && "this assertion is explained here: " \
              "http://eigen.tuxfamily.org/dox-devel/group__TopicUnalignedArrayAssert.html" \
              " **** READ THIS WEB PAGE !!! ****");
#else
  #define EIGEN_MAKE_UNALIGNED_ARRAY_ASSERT(sizemask) \
    eigen_assert((reinterpret_cast<size_t>(array) & sizemask) == 0 \
              && "this assertion is explained here: " \
              "http://eigen.tuxfamily.org/dox-devel/group__TopicUnalignedArrayAssert.html" \
              " **** READ THIS WEB PAGE !!! ****");
#endif

template <typename T, int Size, int MatrixOrArrayOptions>
struct plain_array<T, Size, MatrixOrArrayOptions, 16>
{
  EIGEN_USER_ALIGN16 T array[Size];

  plain_array() 
  { 
    EIGEN_MAKE_UNALIGNED_ARRAY_ASSERT(0xf);
    EIGEN_STATIC_ASSERT(Size * sizeof(T) <= 128 * 128 * 8, OBJECT_ALLOCATED_ON_STACK_IS_TOO_BIG);
  }

  plain_array(constructor_without_unaligned_array_assert) 
  { 
    EIGEN_STATIC_ASSERT(Size * sizeof(T) <= 128 * 128 * 8, OBJECT_ALLOCATED_ON_STACK_IS_TOO_BIG);
  }
};

template <typename T, int MatrixOrArrayOptions, int Alignment>
struct plain_array<T, 0, MatrixOrArrayOptions, Alignment>
{
  EIGEN_USER_ALIGN16 T array[1];
  plain_array() {}
  plain_array(constructor_without_unaligned_array_assert) {}
};

} // end namespace internal

/** \internal
  *
  * \class DenseStorage
  * \ingroup Core_Module
  *
  * \brief Stores the data of a matrix
  *
  * This class stores the data of fixed-size, dynamic-size or mixed matrices
  * in a way as compact as possible.
  *
  * \sa Matrix
  */
template<typename T, int Size, int _Rows, int _Cols, int _Options> class DenseStorage;

// purely fixed-size matrix
template<typename T, int Size, int _Rows, int _Cols, int _Options> class DenseStorage
{
    internal::plain_array<T,Size,_Options> m_data;
  public:
    inline DenseStorage() {}
    inline DenseStorage(internal::constructor_without_unaligned_array_assert)
      : m_data(internal::constructor_without_unaligned_array_assert()) {}
    inline DenseStorage(DenseIndex,DenseIndex,DenseIndex) {}
    inline void swap(DenseStorage& other) { std::swap(m_data,other.m_data); }
    static inline DenseIndex rows(void) {return _Rows;}
    static inline DenseIndex cols(void) {return _Cols;}
    inline void conservativeResize(DenseIndex,DenseIndex,DenseIndex) {}
    inline void resize(DenseIndex,DenseIndex,DenseIndex) {}
    inline const T *data() const { return m_data.array; }
    inline T *data() { return m_data.array; }
};

// null matrix
template<typename T, int _Rows, int _Cols, int _Options> class DenseStorage<T, 0, _Rows, _Cols, _Options>
{
  public:
    inline DenseStorage() {}
    inline DenseStorage(internal::constructor_without_unaligned_array_assert) {}
    inline DenseStorage(DenseIndex,DenseIndex,DenseIndex) {}
    inline void swap(DenseStorage& ) {}
    static inline DenseIndex rows(void) {return _Rows;}
    static inline DenseIndex cols(void) {return _Cols;}
    inline void conservativeResize(DenseIndex,DenseIndex,DenseIndex) {}
    inline void resize(DenseIndex,DenseIndex,DenseIndex) {}
    inline const T *data() const { return 0; }
    inline T *data() { return 0; }
};

// more specializations for null matrices; these are necessary to resolve ambiguities
template<typename T, int _Options> class DenseStorage<T, 0, Dynamic, Dynamic, _Options>
: public DenseStorage<T, 0, 0, 0, _Options> { };

template<typename T, int _Rows, int _Options> class DenseStorage<T, 0, _Rows, Dynamic, _Options>
: public DenseStorage<T, 0, 0, 0, _Options> { };

template<typename T, int _Cols, int _Options> class DenseStorage<T, 0, Dynamic, _Cols, _Options>
: public DenseStorage<T, 0, 0, 0, _Options> { };

// dynamic-size matrix with fixed-size storage
template<typename T, int Size, int _Options> class DenseStorage<T, Size, Dynamic, Dynamic, _Options>
{
    internal::plain_array<T,Size,_Options> m_data;
    DenseIndex m_rows;
    DenseIndex m_cols;
  public:
    inline DenseStorage() : m_rows(0), m_cols(0) {}
    inline DenseStorage(internal::constructor_without_unaligned_array_assert)
      : m_data(internal::constructor_without_unaligned_array_assert()), m_rows(0), m_cols(0) {}
    inline DenseStorage(DenseIndex, DenseIndex nbRows, DenseIndex nbCols) : m_rows(nbRows), m_cols(nbCols) {}
    inline void swap(DenseStorage& other)
    { std::swap(m_data,other.m_data); std::swap(m_rows,other.m_rows); std::swap(m_cols,other.m_cols); }
    inline DenseIndex rows() const {return m_rows;}
    inline DenseIndex cols() const {return m_cols;}
    inline void conservativeResize(DenseIndex, DenseIndex nbRows, DenseIndex nbCols) { m_rows = nbRows; m_cols = nbCols; }
    inline void resize(DenseIndex, DenseIndex nbRows, DenseIndex nbCols) { m_rows = nbRows; m_cols = nbCols; }
    inline const T *data() const { return m_data.array; }
    inline T *data() { return m_data.array; }
};

// dynamic-size matrix with fixed-size storage and fixed width
template<typename T, int Size, int _Cols, int _Options> class DenseStorage<T, Size, Dynamic, _Cols, _Options>
{
    internal::plain_array<T,Size,_Options> m_data;
    DenseIndex m_rows;
  public:
    inline DenseStorage() : m_rows(0) {}
    inline DenseStorage(internal::constructor_without_unaligned_array_assert)
      : m_data(internal::constructor_without_unaligned_array_assert()), m_rows(0) {}
    inline DenseStorage(DenseIndex, DenseIndex nbRows, DenseIndex) : m_rows(nbRows) {}
    inline void swap(DenseStorage& other) { std::swap(m_data,other.m_data); std::swap(m_rows,other.m_rows); }
    inline DenseIndex rows(void) const {return m_rows;}
    inline DenseIndex cols(void) const {return _Cols;}
    inline void conservativeResize(DenseIndex, DenseIndex nbRows, DenseIndex) { m_rows = nbRows; }
    inline void resize(DenseIndex, DenseIndex nbRows, DenseIndex) { m_rows = nbRows; }
    inline const T *data() const { return m_data.array; }
    inline T *data() { return m_data.array; }
};

// dynamic-size matrix with fixed-size storage and fixed height
template<typename T, int Size, int _Rows, int _Options> class DenseStorage<T, Size, _Rows, Dynamic, _Options>
{
    internal::plain_array<T,Size,_Options> m_data;
    DenseIndex m_cols;
  public:
    inline DenseStorage() : m_cols(0) {}
    inline DenseStorage(internal::constructor_without_unaligned_array_assert)
      : m_data(internal::constructor_without_unaligned_array_assert()), m_cols(0) {}
    inline DenseStorage(DenseIndex, DenseIndex, DenseIndex nbCols) : m_cols(nbCols) {}
    inline void swap(DenseStorage& other) { std::swap(m_data,other.m_data); std::swap(m_cols,other.m_cols); }
    inline DenseIndex rows(void) const {return _Rows;}
    inline DenseIndex cols(void) const {return m_cols;}
    inline void conservativeResize(DenseIndex, DenseIndex, DenseIndex nbCols) { m_cols = nbCols; }
    inline void resize(DenseIndex, DenseIndex, DenseIndex nbCols) { m_cols = nbCols; }
    inline const T *data() const { return m_data.array; }
    inline T *data() { return m_data.array; }
};

// purely dynamic matrix.
template<typename T, int _Options> class DenseStorage<T, Dynamic, Dynamic, Dynamic, _Options>
{
    T *m_data;
    DenseIndex m_rows;
    DenseIndex m_cols;
  public:
    inline DenseStorage() : m_data(0), m_rows(0), m_cols(0) {}
    inline DenseStorage(internal::constructor_without_unaligned_array_assert)
       : m_data(0), m_rows(0), m_cols(0) {}
    inline DenseStorage(DenseIndex size, DenseIndex nbRows, DenseIndex nbCols)
      : m_data(internal::conditional_aligned_new_auto<T,(_Options&DontAlign)==0>(size)), m_rows(nbRows), m_cols(nbCols)
    { EIGEN_INTERNAL_DENSE_STORAGE_CTOR_PLUGIN }
    inline ~DenseStorage() { internal::conditional_aligned_delete_auto<T,(_Options&DontAlign)==0>(m_data, m_rows*m_cols); }
    inline void swap(DenseStorage& other)
    { std::swap(m_data,other.m_data); std::swap(m_rows,other.m_rows); std::swap(m_cols,other.m_cols); }
    inline DenseIndex rows(void) const {return m_rows;}
    inline DenseIndex cols(void) const {return m_cols;}
    inline void conservativeResize(DenseIndex size, DenseIndex nbRows, DenseIndex nbCols)
    {
      m_data = internal::conditional_aligned_realloc_new_auto<T,(_Options&DontAlign)==0>(m_data, size, m_rows*m_cols);
      m_rows = nbRows;
      m_cols = nbCols;
    }
    void resize(DenseIndex size, DenseIndex nbRows, DenseIndex nbCols)
    {
      if(size != m_rows*m_cols)
      {
        internal::conditional_aligned_delete_auto<T,(_Options&DontAlign)==0>(m_data, m_rows*m_cols);
        if (size)
          m_data = internal::conditional_aligned_new_auto<T,(_Options&DontAlign)==0>(size);
        else
          m_data = 0;
        EIGEN_INTERNAL_DENSE_STORAGE_CTOR_PLUGIN
      }
      m_rows = nbRows;
      m_cols = nbCols;
    }
    inline const T *data() const { return m_data; }
    inline T *data() { return m_data; }
};

// matrix with dynamic width and fixed height (so that matrix has dynamic size).
template<typename T, int _Rows, int _Options> class DenseStorage<T, Dynamic, _Rows, Dynamic, _Options>
{
    T *m_data;
    DenseIndex m_cols;
  public:
    inline DenseStorage() : m_data(0), m_cols(0) {}
    inline DenseStorage(internal::constructor_without_unaligned_array_assert) : m_data(0), m_cols(0) {}
    inline DenseStorage(DenseIndex size, DenseIndex, DenseIndex nbCols) : m_data(internal::conditional_aligned_new_auto<T,(_Options&DontAlign)==0>(size)), m_cols(nbCols)
    { EIGEN_INTERNAL_DENSE_STORAGE_CTOR_PLUGIN }
    inline ~DenseStorage() { internal::conditional_aligned_delete_auto<T,(_Options&DontAlign)==0>(m_data, _Rows*m_cols); }
    inline void swap(DenseStorage& other) { std::swap(m_data,other.m_data); std::swap(m_cols,other.m_cols); }
    static inline DenseIndex rows(void) {return _Rows;}
    inline DenseIndex cols(void) const {return m_cols;}
    inline void conservativeResize(DenseIndex size, DenseIndex, DenseIndex nbCols)
    {
      m_data = internal::conditional_aligned_realloc_new_auto<T,(_Options&DontAlign)==0>(m_data, size, _Rows*m_cols);
      m_cols = nbCols;
    }
    EIGEN_STRONG_INLINE void resize(DenseIndex size, DenseIndex, DenseIndex nbCols)
    {
      if(size != _Rows*m_cols)
      {
        internal::conditional_aligned_delete_auto<T,(_Options&DontAlign)==0>(m_data, _Rows*m_cols);
        if (size)
          m_data = internal::conditional_aligned_new_auto<T,(_Options&DontAlign)==0>(size);
        else
          m_data = 0;
        EIGEN_INTERNAL_DENSE_STORAGE_CTOR_PLUGIN
      }
      m_cols = nbCols;
    }
    inline const T *data() const { return m_data; }
    inline T *data() { return m_data; }
};

// matrix with dynamic height and fixed width (so that matrix has dynamic size).
template<typename T, int _Cols, int _Options> class DenseStorage<T, Dynamic, Dynamic, _Cols, _Options>
{
    T *m_data;
    DenseIndex m_rows;
  public:
    inline DenseStorage() : m_data(0), m_rows(0) {}
    inline DenseStorage(internal::constructor_without_unaligned_array_assert) : m_data(0), m_rows(0) {}
    inline DenseStorage(DenseIndex size, DenseIndex nbRows, DenseIndex) : m_data(internal::conditional_aligned_new_auto<T,(_Options&DontAlign)==0>(size)), m_rows(nbRows)
    { EIGEN_INTERNAL_DENSE_STORAGE_CTOR_PLUGIN }
    inline ~DenseStorage() { internal::conditional_aligned_delete_auto<T,(_Options&DontAlign)==0>(m_data, _Cols*m_rows); }
    inline void swap(DenseStorage& other) { std::swap(m_data,other.m_data); std::swap(m_rows,other.m_rows); }
    inline DenseIndex rows(void) const {return m_rows;}
    static inline DenseIndex cols(void) {return _Cols;}
    inline void conservativeResize(DenseIndex size, DenseIndex nbRows, DenseIndex)
    {
      m_data = internal::conditional_aligned_realloc_new_auto<T,(_Options&DontAlign)==0>(m_data, size, m_rows*_Cols);
      m_rows = nbRows;
    }
    EIGEN_STRONG_INLINE void resize(DenseIndex size, DenseIndex nbRows, DenseIndex)
    {
      if(size != m_rows*_Cols)
      {
        internal::conditional_aligned_delete_auto<T,(_Options&DontAlign)==0>(m_data, _Cols*m_rows);
        if (size)
          m_data = internal::conditional_aligned_new_auto<T,(_Options&DontAlign)==0>(size);
        else
          m_data = 0;
        EIGEN_INTERNAL_DENSE_STORAGE_CTOR_PLUGIN
      }
      m_rows = nbRows;
    }
    inline const T *data() const { return m_data; }
    inline T *data() { return m_data; }
};

} // end namespace Eigen

#endif // EIGEN_MATRIX_H
