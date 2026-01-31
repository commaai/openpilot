////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2020, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef HSA_RUNTIME_CORE_INC_SDMA_REGISTERS_H_
#define HSA_RUNTIME_CORE_INC_SDMA_REGISTERS_H_

#include <stddef.h>
#include <stdint.h>

namespace rocr {
namespace AMD {

// SDMA packet for VI device.
// Reference: http://people.freedesktop.org/~agd5f/dma_packets.txt

const unsigned int SDMA_OP_COPY = 1;
const unsigned int SDMA_OP_FENCE = 5;
const unsigned int SDMA_OP_TRAP = 6;
const unsigned int SDMA_OP_POLL_REGMEM = 8;
const unsigned int SDMA_OP_ATOMIC = 10;
const unsigned int SDMA_OP_CONST_FILL = 11;
const unsigned int SDMA_OP_TIMESTAMP = 13;
const unsigned int SDMA_OP_GCR = 17;
const unsigned int SDMA_SUBOP_COPY_LINEAR = 0;
const unsigned int SDMA_SUBOP_COPY_LINEAR_RECT = 4;
const unsigned int SDMA_SUBOP_TIMESTAMP_GET_GLOBAL = 2;
const unsigned int SDMA_SUBOP_USER_GCR = 1;
const unsigned int SDMA_ATOMIC_ADD64 = 47;

typedef struct SDMA_PKT_COPY_LINEAR_TAG {
  union {
    struct {
      unsigned int op : 8;
      unsigned int sub_op : 8;
      unsigned int extra_info : 16;
    };
    unsigned int DW_0_DATA;
  } HEADER_UNION;

  union {
    struct {
      unsigned int count : 22;
      unsigned int reserved_0 : 10;
    };
    unsigned int DW_1_DATA;
  } COUNT_UNION;

  union {
    struct {
      unsigned int reserved_0 : 16;
      unsigned int dst_swap : 2;
      unsigned int reserved_1 : 6;
      unsigned int src_swap : 2;
      unsigned int reserved_2 : 6;
    };
    unsigned int DW_2_DATA;
  } PARAMETER_UNION;

  union {
    struct {
      unsigned int src_addr_31_0 : 32;
    };
    unsigned int DW_3_DATA;
  } SRC_ADDR_LO_UNION;

  union {
    struct {
      unsigned int src_addr_63_32 : 32;
    };
    unsigned int DW_4_DATA;
  } SRC_ADDR_HI_UNION;

  union {
    struct {
      unsigned int dst_addr_31_0 : 32;
    };
    unsigned int DW_5_DATA;
  } DST_ADDR_LO_UNION;

  union {
    struct {
      unsigned int dst_addr_63_32 : 32;
    };
    unsigned int DW_6_DATA;
  } DST_ADDR_HI_UNION;

  static const size_t kMaxSize_ = 0x3fffe0;
} SDMA_PKT_COPY_LINEAR;

// linear sub-window
typedef struct SDMA_PKT_COPY_LINEAR_RECT_TAG {
  static const unsigned int pitch_bits = 19;
  static const unsigned int slice_bits = 28;
  static const unsigned int rect_xy_bits = 14;
  static const unsigned int rect_z_bits = 11;

  union {
    struct {
      unsigned int op : 8;
      unsigned int sub_op : 8;
      unsigned int reserved : 13;
      unsigned int element : 3;
    };
    unsigned int DW_0_DATA;
  } HEADER_UNION;

  union {
    struct {
      unsigned int src_addr_31_0 : 32;
    };
    unsigned int DW_1_DATA;
  } SRC_ADDR_LO_UNION;

  union {
    struct {
      unsigned int src_addr_63_32 : 32;
    };
    unsigned int DW_2_DATA;
  } SRC_ADDR_HI_UNION;

  union {
    struct {
      unsigned int src_offset_x : 14;
      unsigned int reserved_1 : 2;
      unsigned int src_offset_y : 14;
      unsigned int reserved_2 : 2;
    };
    unsigned int DW_3_DATA;
  } SRC_PARAMETER_1_UNION;

  union {
    struct {
      unsigned int src_offset_z : 11;
      unsigned int reserved_1 : 2;
      unsigned int src_pitch : pitch_bits;
    };
    unsigned int DW_4_DATA;
  } SRC_PARAMETER_2_UNION;

  union {
    struct {
      unsigned int src_slice_pitch : slice_bits;
      unsigned int reserved_1 : 4;
    };
    unsigned int DW_5_DATA;
  } SRC_PARAMETER_3_UNION;

  union {
    struct {
      unsigned int dst_addr_31_0 : 32;
    };
    unsigned int DW_6_DATA;
  } DST_ADDR_LO_UNION;

  union {
    struct {
      unsigned int dst_addr_63_32 : 32;
    };
    unsigned int DW_7_DATA;
  } DST_ADDR_HI_UNION;

  union {
    struct {
      unsigned int dst_offset_x : 14;
      unsigned int reserved_1 : 2;
      unsigned int dst_offset_y : 14;
      unsigned int reserved_2 : 2;
    };
    unsigned int DW_8_DATA;
  } DST_PARAMETER_1_UNION;

  union {
    struct {
      unsigned int dst_offset_z : 11;
      unsigned int reserved_1 : 2;
      unsigned int dst_pitch : pitch_bits;
    };
    unsigned int DW_9_DATA;
  } DST_PARAMETER_2_UNION;

  union {
    struct {
      unsigned int dst_slice_pitch : slice_bits;
      unsigned int reserved_1 : 4;
    };
    unsigned int DW_10_DATA;
  } DST_PARAMETER_3_UNION;

  union {
    struct {
      unsigned int rect_x : rect_xy_bits;
      unsigned int reserved_1 : 2;
      unsigned int rect_y : rect_xy_bits;
      unsigned int reserved_2 : 2;
    };
    unsigned int DW_11_DATA;
  } RECT_PARAMETER_1_UNION;

  union {
    struct {
      unsigned int rect_z : rect_z_bits;
      unsigned int reserved_1 : 5;
      unsigned int dst_swap : 2;
      unsigned int reserved_2 : 6;
      unsigned int src_swap : 2;
      unsigned int reserved_3 : 6;
    };
    unsigned int DW_12_DATA;
  } RECT_PARAMETER_2_UNION;

} SDMA_PKT_COPY_LINEAR_RECT;

typedef struct SDMA_PKT_CONSTANT_FILL_TAG {
  union {
    struct {
      unsigned int op : 8;
      unsigned int sub_op : 8;
      unsigned int sw : 2;
      unsigned int reserved_0 : 12;
      unsigned int fillsize : 2;
    };
    unsigned int DW_0_DATA;
  } HEADER_UNION;

  union {
    struct {
      unsigned int dst_addr_31_0 : 32;
    };
    unsigned int DW_1_DATA;
  } DST_ADDR_LO_UNION;

  union {
    struct {
      unsigned int dst_addr_63_32 : 32;
    };
    unsigned int DW_2_DATA;
  } DST_ADDR_HI_UNION;

  union {
    struct {
      unsigned int src_data_31_0 : 32;
    };
    unsigned int DW_3_DATA;
  } DATA_UNION;

  union {
    struct {
      unsigned int count : 22;
      unsigned int reserved_0 : 10;
    };
    unsigned int DW_4_DATA;
  } COUNT_UNION;

  static const size_t kMaxSize_ = 0x3fffe0;
} SDMA_PKT_CONSTANT_FILL;

typedef struct SDMA_PKT_FENCE_TAG {
  union {
    struct {
      unsigned int op : 8;
      unsigned int sub_op : 8;
      unsigned int mtype : 3;
      unsigned int gcc : 1;
      unsigned int sys : 1;
      unsigned int pad1 : 1;
      unsigned int snp : 1;
      unsigned int gpa : 1;
      unsigned int l2_policy : 2;
      unsigned int reserved_0 : 6;
    };
    unsigned int DW_0_DATA;
  } HEADER_UNION;

  union {
    struct {
      unsigned int addr_31_0 : 32;
    };
    unsigned int DW_1_DATA;
  } ADDR_LO_UNION;

  union {
    struct {
      unsigned int addr_63_32 : 32;
    };
    unsigned int DW_2_DATA;
  } ADDR_HI_UNION;

  union {
    struct {
      unsigned int data : 32;
    };
    unsigned int DW_3_DATA;
  } DATA_UNION;
} SDMA_PKT_FENCE;

typedef struct SDMA_PKT_POLL_REGMEM_TAG {
  union {
    struct {
      unsigned int op : 8;
      unsigned int sub_op : 8;
      unsigned int reserved_0 : 10;
      unsigned int hdp_flush : 1;
      unsigned int reserved_1 : 1;
      unsigned int func : 3;
      unsigned int mem_poll : 1;
    };
    unsigned int DW_0_DATA;
  } HEADER_UNION;

  union {
    struct {
      unsigned int addr_31_0 : 32;
    };
    unsigned int DW_1_DATA;
  } ADDR_LO_UNION;

  union {
    struct {
      unsigned int addr_63_32 : 32;
    };
    unsigned int DW_2_DATA;
  } ADDR_HI_UNION;

  union {
    struct {
      unsigned int value : 32;
    };
    unsigned int DW_3_DATA;
  } VALUE_UNION;

  union {
    struct {
      unsigned int mask : 32;
    };
    unsigned int DW_4_DATA;
  } MASK_UNION;

  union {
    struct {
      unsigned int interval : 16;
      unsigned int retry_count : 12;
      unsigned int reserved_0 : 4;
    };
    unsigned int DW_5_DATA;
  } DW5_UNION;
} SDMA_PKT_POLL_REGMEM;

typedef struct SDMA_PKT_ATOMIC_TAG {
  union {
    struct {
      unsigned int op : 8;
      unsigned int sub_op : 8;
      unsigned int l : 1;
      unsigned int reserved_0 : 8;
      unsigned int operation : 7;
    };
    unsigned int DW_0_DATA;
  } HEADER_UNION;

  union {
    struct {
      unsigned int addr_31_0 : 32;
    };
    unsigned int DW_1_DATA;
  } ADDR_LO_UNION;

  union {
    struct {
      unsigned int addr_63_32 : 32;
    };
    unsigned int DW_2_DATA;
  } ADDR_HI_UNION;

  union {
    struct {
      unsigned int src_data_31_0 : 32;
    };
    unsigned int DW_3_DATA;
  } SRC_DATA_LO_UNION;

  union {
    struct {
      unsigned int src_data_63_32 : 32;
    };
    unsigned int DW_4_DATA;
  } SRC_DATA_HI_UNION;

  union {
    struct {
      unsigned int cmp_data_31_0 : 32;
    };
    unsigned int DW_5_DATA;
  } CMP_DATA_LO_UNION;

  union {
    struct {
      unsigned int cmp_data_63_32 : 32;
    };
    unsigned int DW_6_DATA;
  } CMP_DATA_HI_UNION;

  union {
    struct {
      unsigned int loop_interval : 13;
      unsigned int reserved_0 : 19;
    };
    unsigned int DW_7_DATA;
  } LOOP_UNION;
} SDMA_PKT_ATOMIC;

typedef struct SDMA_PKT_TIMESTAMP_TAG {
  union {
    struct {
      unsigned int op : 8;
      unsigned int sub_op : 8;
      unsigned int reserved_0 : 16;
    };
    unsigned int DW_0_DATA;
  } HEADER_UNION;

  union {
    struct {
      unsigned int addr_31_0 : 32;
    };
    unsigned int DW_1_DATA;
  } ADDR_LO_UNION;

  union {
    struct {
      unsigned int addr_63_32 : 32;
    };
    unsigned int DW_2_DATA;
  } ADDR_HI_UNION;

} SDMA_PKT_TIMESTAMP;

typedef struct SDMA_PKT_TRAP_TAG {
  union {
    struct {
      unsigned int op : 8;
      unsigned int sub_op : 8;
      unsigned int reserved_0 : 16;
    };
    unsigned int DW_0_DATA;
  } HEADER_UNION;

  union {
    struct {
      unsigned int int_ctx : 28;
      unsigned int reserved_1 : 4;
    };
    unsigned int DW_1_DATA;
  } INT_CONTEXT_UNION;
} SDMA_PKT_TRAP;

// HDP flush packet, no parameters.
typedef struct SDMA_PKT_HDP_FLUSH_TAG {
  unsigned int DW_0_DATA;
  unsigned int DW_1_DATA;
  unsigned int DW_2_DATA;
  unsigned int DW_3_DATA;
  unsigned int DW_4_DATA;
  unsigned int DW_5_DATA;

  // Version of gfx9 sDMA microcode introducing SDMA_PKT_HDP_FLUSH
  static const uint16_t kMinVersion_ = 0x1A5;
} SDMA_PKT_HDP_FLUSH;
static const SDMA_PKT_HDP_FLUSH hdp_flush_cmd = {0x8, 0x0, 0x80000000, 0x0, 0x0, 0x0};

typedef struct SDMA_PKT_GCR_TAG {
  union {
    struct {
      unsigned int op : 8;
      unsigned int sub_op : 8;
      unsigned int : 16;
    };
    unsigned int DW_0_DATA;
  } HEADER_UNION;

  union {
    struct {
      unsigned int : 7;
      unsigned int BaseVA_LO : 25;
    };
    unsigned int DW_1_DATA;
  } WORD1_UNION;

  union {
    struct {
      unsigned int BaseVA_HI : 16;
      unsigned int GCR_CONTROL_GLI_INV : 2;
      unsigned int GCR_CONTROL_GL1_RANGE : 2;
      unsigned int GCR_CONTROL_GLM_WB : 1;
      unsigned int GCR_CONTROL_GLM_INV : 1;
      unsigned int GCR_CONTROL_GLK_WB : 1;
      unsigned int GCR_CONTROL_GLK_INV : 1;
      unsigned int GCR_CONTROL_GLV_INV : 1;
      unsigned int GCR_CONTROL_GL1_INV : 1;
      unsigned int GCR_CONTROL_GL2_US : 1;
      unsigned int GCR_CONTROL_GL2_RANGE : 2;
      unsigned int GCR_CONTROL_GL2_DISCARD : 1;
      unsigned int GCR_CONTROL_GL2_INV : 1;
      unsigned int GCR_CONTROL_GL2_WB : 1;
    };
    unsigned int DW_2_DATA;
  } WORD2_UNION;

  union {
    struct {
      unsigned int GCR_CONTROL_RANGE_IS_PA : 1;
      unsigned int GCR_CONTROL_SEQ : 2;
      unsigned int : 4;
      unsigned int LimitVA_LO : 25;
    };
    unsigned int DW_3_DATA;
  } WORD3_UNION;

  union {
    struct {
      unsigned int LimitVA_HI : 16;
      unsigned int : 8;
      unsigned int VMID : 4;
      unsigned int : 4;
    };
    unsigned int DW_4_DATA;
  } WORD4_UNION;
} SDMA_PKT_GCR;

}  // namespace amd
}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_INC_SDMA_REGISTERS_H_
