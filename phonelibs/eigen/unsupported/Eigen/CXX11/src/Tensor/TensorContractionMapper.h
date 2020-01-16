// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_CONTRACTION_MAPPER_H
#define EIGEN_CXX11_TENSOR_TENSOR_CONTRACTION_MAPPER_H

namespace Eigen {

namespace internal {

enum {
  Rhs = 0,
  Lhs = 1
};

/*
 * Implementation of the Eigen blas_data_mapper class for tensors.
 */

template <typename Tensor, bool HasRawAccess> struct CoeffLoader {
  enum {
    DirectOffsets = false
  };

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE CoeffLoader(const Tensor& tensor) : m_tensor(tensor) { }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void offsetBuffer(typename Tensor::Index) {
    eigen_assert(false && "unsupported");
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE typename Tensor::Scalar coeff(typename Tensor::Index index) const { return m_tensor.coeff(index); }

 template<int LoadMode> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
 typename Tensor::PacketReturnType packet(typename Tensor::Index index) const
  {
    return m_tensor.template packet<LoadMode>(index);
  }


 private:
  const Tensor m_tensor;
};

template <typename Tensor> struct CoeffLoader<Tensor, true> {
  enum {
    DirectOffsets = true
  };

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE CoeffLoader(const Tensor& tensor) : m_data(tensor.data()) {}

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void offsetBuffer(typename Tensor::Index offset) {
    m_data += offset;
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE typename Tensor::Scalar coeff(typename Tensor::Index index) const { return loadConstant(m_data+index); }

 template<int LoadMode> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
 typename Tensor::PacketReturnType packet(typename Tensor::Index index) const
  {
    return internal::ploadt_ro<typename Tensor::PacketReturnType, LoadMode>(m_data + index);
  }
 private:
  typedef typename Tensor::Scalar Scalar;
  const Scalar* m_data;
};

template<typename Scalar, typename Index, int side,
         typename Tensor,
         typename nocontract_t, typename contract_t,
         int packet_size, bool inner_dim_contiguous, int Alignment>
class SimpleTensorContractionMapper {
  public:
  EIGEN_DEVICE_FUNC
  SimpleTensorContractionMapper(const Tensor& tensor,
                                const nocontract_t& nocontract_strides,
                                const nocontract_t& ij_strides,
                                const contract_t& contract_strides,
                                const contract_t& k_strides) :
      m_tensor(tensor),
      m_nocontract_strides(nocontract_strides),
      m_ij_strides(ij_strides),
      m_contract_strides(contract_strides),
      m_k_strides(k_strides) { }

  enum {
    DirectOffsets = CoeffLoader<Tensor, Tensor::RawAccess>::DirectOffsets
  };

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void offsetBuffer(typename Tensor::Index offset) {
    m_tensor.offsetBuffer(offset);
  }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE void prefetch(Index /*i*/) { }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Scalar operator()(Index row) const {
    // column major assumption
    return operator()(row, 0);
  }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Scalar operator()(Index row, Index col) const {
    return m_tensor.coeff(computeIndex(row, col));
  }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Index computeIndex(Index row, Index col) const {
    const bool left = (side == Lhs);
    Index nocontract_val = left ? row : col;
    Index linidx = 0;
    for (int i = static_cast<int>(array_size<nocontract_t>::value) - 1; i > 0; i--) {
      const Index idx = nocontract_val / m_ij_strides[i];
      linidx += idx * m_nocontract_strides[i];
      nocontract_val -= idx * m_ij_strides[i];
    }
    if (array_size<typename Tensor::Dimensions>::value > array_size<contract_t>::value) {
      if (side == Lhs && inner_dim_contiguous) {
        eigen_assert(m_nocontract_strides[0] == 1);
        linidx += nocontract_val;
      } else {
        linidx += nocontract_val * m_nocontract_strides[0];
      }
    }

    Index contract_val = left ? col : row;
    if(array_size<contract_t>::value > 0) {
      for (int i = static_cast<int>(array_size<contract_t>::value) - 1; i > 0; i--) {
        const Index idx = contract_val / m_k_strides[i];
        linidx += idx * m_contract_strides[i];
        contract_val -= idx * m_k_strides[i];
      }

      if (side == Rhs && inner_dim_contiguous) {
        eigen_assert(m_contract_strides[0] == 1);
        linidx += contract_val;
      } else {
        linidx += contract_val * m_contract_strides[0];
      }
    }

    return linidx;
  }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE IndexPair<Index> computeIndexPair(Index row, Index col, const Index distance) const {
    const bool left = (side == Lhs);
    Index nocontract_val[2] = {left ? row : col, left ? row + distance : col};
    Index linidx[2] = {0, 0};
    if (array_size<typename Tensor::Dimensions>::value > array_size<contract_t>::value) {
      for (int i = static_cast<int>(array_size<nocontract_t>::value) - 1; i > 0; i--) {
        const Index idx0 = nocontract_val[0] / m_ij_strides[i];
        const Index idx1 = nocontract_val[1] / m_ij_strides[i];
        linidx[0] += idx0 * m_nocontract_strides[i];
        linidx[1] += idx1 * m_nocontract_strides[i];
        nocontract_val[0] -= idx0 * m_ij_strides[i];
        nocontract_val[1] -= idx1 * m_ij_strides[i];
      }
      if (side == Lhs && inner_dim_contiguous) {
        eigen_assert(m_nocontract_strides[0] == 1);
        linidx[0] += nocontract_val[0];
        linidx[1] += nocontract_val[1];
      } else {
        linidx[0] += nocontract_val[0] * m_nocontract_strides[0];
        linidx[1] += nocontract_val[1] * m_nocontract_strides[0];
      }
    }

    Index contract_val[2] = {left ? col : row, left ? col : row + distance};
    if (array_size<contract_t>::value> 0) {
      for (int i = static_cast<int>(array_size<contract_t>::value) - 1; i > 0; i--) {
        const Index idx0 = contract_val[0] / m_k_strides[i];
        const Index idx1 = contract_val[1] / m_k_strides[i];
        linidx[0] += idx0 * m_contract_strides[i];
        linidx[1] += idx1 * m_contract_strides[i];
        contract_val[0] -= idx0 * m_k_strides[i];
        contract_val[1] -= idx1 * m_k_strides[i];
      }

      if (side == Rhs && inner_dim_contiguous) {
        eigen_assert(m_contract_strides[0] == 1);
        linidx[0] += contract_val[0];
        linidx[1] += contract_val[1];
      } else {
        linidx[0] += contract_val[0] * m_contract_strides[0];
        linidx[1] += contract_val[1] * m_contract_strides[0];
      }
    }
    return IndexPair<Index>(linidx[0], linidx[1]);
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Index firstAligned(Index size) const {
    // Only claim alignment when we can compute the actual stride (ie when we're
    // dealing with the lhs with inner_dim_contiguous. This is because the
    // matrix-vector product relies on the stride when dealing with aligned inputs.
    return (Alignment == Aligned) && (side == Lhs) && inner_dim_contiguous ? 0 : size;
  }
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Index stride() const {
    return ((side == Lhs) && inner_dim_contiguous && array_size<contract_t>::value > 0) ? m_contract_strides[0] : 1;
  }

 protected:
  CoeffLoader<Tensor, Tensor::RawAccess> m_tensor;
  const nocontract_t m_nocontract_strides;
  const nocontract_t m_ij_strides;
  const contract_t m_contract_strides;
  const contract_t m_k_strides;
};


template<typename Scalar, typename Index, int side,
         typename Tensor,
         typename nocontract_t, typename contract_t,
         int packet_size, bool inner_dim_contiguous,
         bool inner_dim_reordered, int Alignment>
class BaseTensorContractionMapper : public SimpleTensorContractionMapper<Scalar, Index, side, Tensor, nocontract_t, contract_t, packet_size, inner_dim_contiguous, Alignment>
{
 public:
  typedef SimpleTensorContractionMapper<Scalar, Index, side, Tensor, nocontract_t, contract_t, packet_size, inner_dim_contiguous, Alignment> ParentMapper;

  EIGEN_DEVICE_FUNC
  BaseTensorContractionMapper(const Tensor& tensor,
                              const nocontract_t& nocontract_strides,
                              const nocontract_t& ij_strides,
                              const contract_t& contract_strides,
                              const contract_t& k_strides) :
  ParentMapper(tensor, nocontract_strides, ij_strides, contract_strides, k_strides) { }

  typedef typename Tensor::PacketReturnType Packet;
  typedef typename unpacket_traits<Packet>::half HalfPacket;

  template <int AlignmentType>
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Packet loadPacket(Index i, Index j) const {
    // whole method makes column major assumption

    // don't need to add offsets for now (because operator handles that)
    // current code assumes packet size must be a multiple of 2
    EIGEN_STATIC_ASSERT(packet_size % 2 == 0, YOU_MADE_A_PROGRAMMING_MISTAKE);

    if (Tensor::PacketAccess && inner_dim_contiguous && !inner_dim_reordered) {
      const Index index = this->computeIndex(i, j);
      eigen_assert(this->computeIndex(i+packet_size-1, j) == index + packet_size-1);
      return this->m_tensor.template packet<AlignmentType>(index);
    }

    const IndexPair<Index> indexPair = this->computeIndexPair(i, j, packet_size - 1);
    const Index first = indexPair.first;
    const Index last = indexPair.second;

    // We can always do optimized packet reads from left hand side right now, because
    // the vertical matrix dimension on the left hand side is never contracting.
    // On the right hand side we need to check if the contracting dimensions may have
    // been shuffled first.
    if (Tensor::PacketAccess &&
        (side == Lhs || internal::array_size<contract_t>::value <= 1 || !inner_dim_reordered) &&
        (last - first) == (packet_size - 1)) {

      return this->m_tensor.template packet<AlignmentType>(first);
    }

    EIGEN_ALIGN_MAX Scalar data[packet_size];

    data[0] = this->m_tensor.coeff(first);
    for (Index k = 1; k < packet_size - 1; k += 2) {
      const IndexPair<Index> internal_pair = this->computeIndexPair(i + k, j, 1);
      data[k] = this->m_tensor.coeff(internal_pair.first);
      data[k + 1] = this->m_tensor.coeff(internal_pair.second);
    }
    data[packet_size - 1] = this->m_tensor.coeff(last);

    return pload<Packet>(data);
  }

  template <int AlignmentType>
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE HalfPacket loadHalfPacket(Index i, Index j) const {
    // whole method makes column major assumption

    // don't need to add offsets for now (because operator handles that)
    const Index half_packet_size = unpacket_traits<HalfPacket>::size;
    if (half_packet_size == packet_size) {
      return loadPacket<AlignmentType>(i, j);
    }
    EIGEN_ALIGN_MAX Scalar data[half_packet_size];
    for (Index k = 0; k < half_packet_size; k++) {
      data[k] = operator()(i + k, j);
    }
    return pload<HalfPacket>(data);
  }
};


template<typename Scalar, typename Index, int side,
         typename Tensor,
         typename nocontract_t, typename contract_t,
         bool inner_dim_contiguous,
         bool inner_dim_reordered, int Alignment>
class BaseTensorContractionMapper<Scalar, Index, side, Tensor, nocontract_t, contract_t, 1, inner_dim_contiguous, inner_dim_reordered, Alignment> : public SimpleTensorContractionMapper<Scalar, Index, side, Tensor, nocontract_t, contract_t, 1, inner_dim_contiguous, Alignment>
{
 public:
  typedef SimpleTensorContractionMapper<Scalar, Index, side, Tensor, nocontract_t, contract_t, 1, inner_dim_contiguous, Alignment> ParentMapper;

  EIGEN_DEVICE_FUNC
  BaseTensorContractionMapper(const Tensor& tensor,
                              const nocontract_t& nocontract_strides,
                              const nocontract_t& ij_strides,
                              const contract_t& contract_strides,
                              const contract_t& k_strides) :
  ParentMapper(tensor, nocontract_strides, ij_strides, contract_strides, k_strides) { }

  typedef typename Tensor::PacketReturnType Packet;
  template <int> EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Packet loadPacket(Index i, Index j) const {
    EIGEN_ALIGN_MAX Scalar data[1];
    data[0] = this->m_tensor.coeff(this->computeIndex(i, j));
    return pload<typename Tensor::PacketReturnType>(data);
  }
  template <int> EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Packet loadHalfPacket(Index i, Index j) const {
    return loadPacket(i, j);
  }
};


template<typename Scalar, typename Index, int side,
         typename Tensor,
         typename nocontract_t, typename contract_t,
         int packet_size,
         bool inner_dim_contiguous, bool inner_dim_reordered, int Alignment>
class TensorContractionSubMapper {
 public:
  typedef typename Tensor::PacketReturnType Packet;
  typedef typename unpacket_traits<Packet>::half HalfPacket;

  typedef BaseTensorContractionMapper<Scalar, Index, side, Tensor, nocontract_t, contract_t, packet_size, inner_dim_contiguous, inner_dim_reordered, Alignment> ParentMapper;
  typedef TensorContractionSubMapper<Scalar, Index, side, Tensor, nocontract_t, contract_t, packet_size, inner_dim_contiguous, inner_dim_reordered, Alignment> Self;
  typedef Self LinearMapper;

  enum {
    // We can use direct offsets iff the parent mapper supports then and we can compute the strides.
    // TODO: we should also enable direct offsets for the Rhs case.
    UseDirectOffsets = ParentMapper::DirectOffsets && (side == Lhs) && inner_dim_contiguous && (array_size<contract_t>::value > 0)
  };

  EIGEN_DEVICE_FUNC TensorContractionSubMapper(const ParentMapper& base_mapper, Index vert_offset, Index horiz_offset)
      : m_base_mapper(base_mapper), m_vert_offset(vert_offset), m_horiz_offset(horiz_offset) {
    // Bake the offsets into the buffer used by the base mapper whenever possible. This avoids the need to recompute
    // this offset every time we attempt to access a coefficient.
    if (UseDirectOffsets) {
      Index stride = m_base_mapper.stride();
      m_base_mapper.offsetBuffer(vert_offset + horiz_offset * stride);
    }
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Scalar operator()(Index i) const {
    if (UseDirectOffsets) {
      return m_base_mapper(i, 0);
    }
    return m_base_mapper(i + m_vert_offset, m_horiz_offset);
  }
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Scalar operator()(Index i, Index j) const {
    if (UseDirectOffsets) {
      return m_base_mapper(i, j);
    }
    return m_base_mapper(i + m_vert_offset, j + m_horiz_offset);
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Packet loadPacket(Index i) const {
    if (UseDirectOffsets) {
      return m_base_mapper.template loadPacket<Alignment>(i, 0);
    }
    return m_base_mapper.template loadPacket<Alignment>(i + m_vert_offset, m_horiz_offset);
  }
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Packet loadPacket(Index i, Index j) const {
    if (UseDirectOffsets) {
      return m_base_mapper.template loadPacket<Alignment>(i, j);
    }
    return m_base_mapper.template loadPacket<Alignment>(i + m_vert_offset, j + m_horiz_offset);
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE HalfPacket loadHalfPacket(Index i) const {
    if (UseDirectOffsets) {
      return m_base_mapper.template loadHalfPacket<Alignment>(i, 0);
    }
    return m_base_mapper.template loadHalfPacket<Alignment>(i + m_vert_offset, m_horiz_offset);
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void storePacket(Index i, Packet p) const {
    if (UseDirectOffsets) {
      m_base_mapper.storePacket(i, 0, p);
    }
    m_base_mapper.storePacket(i + m_vert_offset, m_horiz_offset, p);
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE LinearMapper getLinearMapper(Index i, Index j) const {
    if (UseDirectOffsets) {
      return LinearMapper(m_base_mapper, i, j);
    }
    return LinearMapper(m_base_mapper, i + m_vert_offset, j + m_horiz_offset);
  }

  template <typename PacketT, int AlignmentType>
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE PacketT load(Index i) const {
    EIGEN_STATIC_ASSERT((internal::is_same<PacketT, Packet>::value), YOU_MADE_A_PROGRAMMING_MISTAKE);
    const int ActualAlignment = (AlignmentType == Aligned) && (Alignment == Aligned) ? Aligned : Unaligned;
    if (UseDirectOffsets) {
     return m_base_mapper.template loadPacket<ActualAlignment>(i, 0);
    }
    return m_base_mapper.template loadPacket<ActualAlignment>(i + m_vert_offset, m_horiz_offset);
  }

  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE bool aligned(Index) const {
    return false;
  }

 private:
  ParentMapper m_base_mapper;
  const Index m_vert_offset;
  const Index m_horiz_offset;
};


template<typename Scalar_, typename Index, int side,
         typename Tensor,
         typename nocontract_t, typename contract_t,
         int packet_size,
         bool inner_dim_contiguous, bool inner_dim_reordered, int Alignment>
class TensorContractionInputMapper
  : public BaseTensorContractionMapper<Scalar_, Index, side, Tensor, nocontract_t, contract_t, packet_size, inner_dim_contiguous, inner_dim_reordered, Alignment> {

 public:
  typedef Scalar_ Scalar;
  typedef BaseTensorContractionMapper<Scalar, Index, side, Tensor, nocontract_t, contract_t, packet_size, inner_dim_contiguous, inner_dim_reordered, Alignment> Base;
  typedef TensorContractionSubMapper<Scalar, Index, side, Tensor, nocontract_t, contract_t, packet_size, inner_dim_contiguous, inner_dim_reordered, Alignment> SubMapper;
  typedef SubMapper VectorMapper;

  EIGEN_DEVICE_FUNC TensorContractionInputMapper(const Tensor& tensor,
                               const nocontract_t& nocontract_strides,
                               const nocontract_t& ij_strides,
                               const contract_t& contract_strides,
                               const contract_t& k_strides)
      : Base(tensor, nocontract_strides, ij_strides, contract_strides, k_strides) { }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE SubMapper getSubMapper(Index i, Index j) const {
    return SubMapper(*this, i, j);
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE VectorMapper getVectorMapper(Index i, Index j) const {
    return VectorMapper(*this, i, j);
  }
};



}  // end namespace internal
}  // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_CONTRACTION_MAPPER_H
