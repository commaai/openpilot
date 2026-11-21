/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef __NVDEC_DRV_H_
#define __NVDEC_DRV_H_

// TODO: Many fields can be converted to bitfields to save memory BW
// TODO: Revisit reserved fields for proper alignment and memory savings

///////////////////////////////////////////////////////////////////////////////
// NVDEC(MSDEC 5) is a single engine solution, and seperates into VLD, MV, IQT,
//                MCFETCH, MC, MCC, REC, DBF, DFBFDMA, HIST etc unit.
//                The class(driver to HW) can mainly seperate into VLD parser
//                and Decoder part to be consistent with original design. And
//                the sequence level info usally set in VLD part. Later codec like
//                VP8 won't name in this way.
// MSVLD: Multi-Standard VLD parser.
//
#define ALIGN_UP(v, n)          (((v) + ((n)-1)) &~ ((n)-1))
#define NVDEC_ALIGN(value)      ALIGN_UP(value,256) // Align to 256 bytes
#define NVDEC_MAX_MPEG2_SLICE   65536 // at 4096*4096, macroblock count = 65536, 1 macroblock per slice

#define NVDEC_CODEC_MPEG1   0
#define NVDEC_CODEC_MPEG2   1
#define NVDEC_CODEC_VC1     2
#define NVDEC_CODEC_H264    3
#define NVDEC_CODEC_MPEG4   4
#define NVDEC_CODEC_DIVX    NVDEC_CODEC_MPEG4
#define NVDEC_CODEC_VP8     5
#define NVDEC_CODEC_HEVC    7
#define NVDEC_CODEC_VP9     9
#define NVDEC_CODEC_HEVC_PARSER 12
#define NVDEC_CODEC_AV1         10

// AES encryption
enum
{
    AES128_NONE = 0x0,
    AES128_CTR = 0x1,
    AES128_CBC,
    AES128_ECB,
    AES128_OFB,
    AES128_CTR_LSB16B,
    AES128_CLR_AS_ENCRYPT,
    AES128_RESERVED = 0x7
};

enum
{
    AES128_CTS_DISABLE = 0x0,
    AES128_CTS_ENABLE = 0x1
};

enum
{
    AES128_PADDING_NONE = 0x0,
    AES128_PADDING_CARRY_OVER,
    AES128_PADDING_RFC2630,
    AES128_PADDING_RESERVED = 0x7
};

typedef enum
{
    ENCR_MODE_CTR64         = 0,
    ENCR_MODE_CBC           = 1,
    ENCR_MODE_ECB           = 2,
    ENCR_MODE_ECB_PARTIAL   = 3,
    ENCR_MODE_CBC_PARTIAL   = 4,
    ENCR_MODE_CLEAR_INTO_VPR = 5,     // used for clear stream decoding into VPR.
    ENCR_MODE_FORCE_INTO_VPR = 6,    //  used to force decode output into VPR.
} ENCR_MODE;

// drm_mode configuration
//
// Bit 0:2  AES encryption mode
// Bit 3    CTS (CipherTextStealing) enable/disable
// Bit 4:6  Padding type
// Bit 7:7  Unwrap key enable/disable

#define AES_MODE_MASK           0x7
#define AES_CTS_MASK            0x1
#define AES_PADDING_TYPE_MASK   0x7
#define AES_UNWRAP_KEY_MASK     0x1

#define AES_MODE_SHIFT          0
#define AES_CTS_SHIFT           3
#define AES_PADDING_TYPE_SHIFT  4
#define AES_UNWRAP_KEY_SHIFT    7

#define AES_SET_FLAG(M, C, P)   ((M & AES_MODE_MASK) << AES_MODE_SHIFT) | \
                                ((C & AES_CTS_MASK) << AES_CTS_SHIFT) | \
                                ((P & AES_PADDING_TYPE_MASK) << AES_PADDING_TYPE_SHIFT)

#define AES_GET_FLAG(V, F)      ((V & ((AES_##F##_MASK) <<(AES_##F##_SHIFT))) >> (AES_##F##_SHIFT))

#define DRM_MODE_MASK           0x7f        // Bits 0:6  (0:2 -> AES_MODE, 3 -> AES_CTS, 4:6 -> AES_PADDING_TYPE)
#define AES_GET_DRM_MODE(V)      (V & DRM_MODE_MASK)

enum { DRM_MS_PIFF_CTR  =   AES_SET_FLAG(AES128_CTR, AES128_CTS_DISABLE, AES128_PADDING_CARRY_OVER) };
enum { DRM_MS_PIFF_CBC  =   AES_SET_FLAG(AES128_CBC, AES128_CTS_DISABLE, AES128_PADDING_NONE) };
enum { DRM_MARLIN_CTR   =   AES_SET_FLAG(AES128_CTR, AES128_CTS_DISABLE, AES128_PADDING_NONE) };
enum { DRM_MARLIN_CBC   =   AES_SET_FLAG(AES128_CBC, AES128_CTS_DISABLE, AES128_PADDING_RFC2630) };
enum { DRM_WIDEVINE     =   AES_SET_FLAG(AES128_CBC, AES128_CTS_ENABLE,  AES128_PADDING_NONE) };
enum { DRM_WIDEVINE_CTR =   AES_SET_FLAG(AES128_CTR, AES128_CTS_DISABLE, AES128_PADDING_CARRY_OVER) };
enum { DRM_ULTRA_VIOLET =   AES_SET_FLAG(AES128_CTR_LSB16B, AES128_CTS_DISABLE, AES128_PADDING_NONE) };
enum { DRM_NONE         =   AES_SET_FLAG(AES128_NONE, AES128_CTS_DISABLE, AES128_PADDING_NONE) };
enum { DRM_CLR_AS_ENCRYPT = AES_SET_FLAG(AES128_CLR_AS_ENCRYPT, AES128_CTS_DISABLE, AES128_PADDING_NONE)};

// SSM entry structure
typedef struct _nvdec_ssm_s {
    unsigned int bytes_of_protected_data;//bytes of protected data, follows bytes_of_clear_data. Note: When padding is enabled, it does not include the padding_bytes (1~15), which can be derived by "(16-(bytes_of_protected_data&0xF))&0xF"
    unsigned int bytes_of_clear_data:16; //bytes of clear data, located before bytes_of_protected_data
    unsigned int skip_byte_blk      : 4; //valid when (entry_type==0 && mode = 1)
    unsigned int crypt_byte_blk     : 4; //valid when (entry_type==0 && mode = 1)
    unsigned int skip               : 1; //whether this SSM entry should be skipped or not
    unsigned int last               : 1; //whether this SSM entry is the last one for the whole decoding frame
    unsigned int pad                : 1; //valid when (entry_type==0 && mode==0 && AES_PADDING_TYPE==AES128_PADDING_RFC2630), 0 for pad_end, 1 for pad_begin
    unsigned int mode               : 1; //0 for normal mode, 1 for pattern mode
    unsigned int entry_type         : 1; //0 for DATA, 1 for IV
    unsigned int reserved           : 3;
} nvdec_ssm_s; /* SubSampleMap, 8bytes */

// PASS2 OTF extension structure for SSM support, not exist in nvdec_mpeg4_pic_s (as MPEG4 OTF SW-DRM is not supported yet)
typedef struct _nvdec_pass2_otf_ext_s {
    unsigned int ssm_entry_num      :16; //specifies how many SSM entries (each in unit of 8 bytes) existed in SET_SUB_SAMPLE_MAP_OFFSET surface
    unsigned int ssm_iv_num         :16; //specifies how many SSM IV (each in unit of 16 bytes) existed in SET_SUB_SAMPLE_MAP_IV_OFFSET surface
    unsigned int real_stream_length;     //the real stream length, which is the bitstream length EMD/VLD will get after whole frame SSM processing, sum up of "clear+protected" bytes in SSM entries and removing "non_slice_data/skip".
    unsigned int non_slice_data     :16; //specifies the first many bytes needed to skip, includes only those of "clear+protected" bytes ("padding" bytes excluded)
    unsigned int drm_mode           : 7;
    unsigned int reserved           : 9;
} nvdec_pass2_otf_ext_s; /* 12bytes */


//NVDEC5.0 low latency decoding (partial stream kickoff without context switch), method will reuse HevcSetSliceInfoBufferOffset.
typedef struct _nvdec_substream_entry_s {
    unsigned int substream_start_offset;                    //substream byte start offset to bitstream base address
    unsigned int substream_length;                          //subsream length in byte  
    unsigned int substream_first_tile_idx           : 8;    //the first tile index(raster scan in frame) of this substream,max is 255 
    unsigned int substream_last_tile_idx            : 8;    //the last tile index(raster scan in frame) of this substream, max is 255
    unsigned int last_substream_entry_in_frame      : 1;    //this entry is the last substream entry of this frame
    unsigned int reserved                           : 15;
} nvdec_substream_entry_s;/*low latency without context switch substream entry map,12bytes*/


// GIP

/* tile border coefficients of filter */
#define GIP_ASIC_VERT_FILTER_RAM_SIZE       16  /* bytes per pixel */

/* BSD control data of current picture at tile border
 * 11  * 128 bits per 4x4 tile = 128/(8*4) bytes per row */
#define GIP_ASIC_BSD_CTRL_RAM_SIZE          4  /* bytes per row */

/* 8 dc + 8 to boundary + 6*16 + 2*6*64 + 2*64 -> 63 * 16 bytes */
#define GIP_ASIC_SCALING_LIST_SIZE          (16*64)

/* tile border coefficients of filter */
#define GIP_ASIC_VERT_SAO_RAM_SIZE          16  /* bytes per pixel */

/* max number of tiles times width and height (2 bytes each),
 * rounding up to next 16 bytes boundary + one extra 16 byte
 * chunk (HW guys wanted to have this) */
#define GIP_ASIC_TILE_SIZE                  ((20*22*2*2+16+15) & ~0xF)

/* Segment map uses 32 bytes / CTB */
#define GIP_ASIC_VP9_CTB_SEG_SIZE           32

// HEVC Filter FG buffer
#define HEVC_DBLK_TOP_SIZE_IN_SB16          ALIGN_UP(632, 128) // ctb16 + 444
#define HEVC_DBLK_TOP_BUF_SIZE(w)           NVDEC_ALIGN( (ALIGN_UP(w,16)/16 + 2) * HEVC_DBLK_TOP_SIZE_IN_SB16) // 8K: 1285*256

#define HEVC_DBLK_LEFT_SIZE_IN_SB16         ALIGN_UP(506, 128) // ctb16 + 444
#define HEVC_DBLK_LEFT_BUF_SIZE(h)          NVDEC_ALIGN( (ALIGN_UP(h,16)/16 + 2) * HEVC_DBLK_LEFT_SIZE_IN_SB16) // 8K: 1028*256

#define HEVC_SAO_LEFT_SIZE_IN_SB16          ALIGN_UP(713, 128) // ctb16 + 444
#define HEVC_SAO_LEFT_BUF_SIZE(h)           NVDEC_ALIGN( (ALIGN_UP(h,16)/16 + 2) * HEVC_SAO_LEFT_SIZE_IN_SB16) // 8K: 1542*256

// VP9 Filter FG buffer
#define VP9_DBLK_TOP_SIZE_IN_SB64           ALIGN_UP(2000, 128) // 420
#define VP9_DBLK_TOP_BUF_SIZE(w)            NVDEC_ALIGN( (ALIGN_UP(w,64)/64 + 2) * VP9_DBLK_TOP_SIZE_IN_SB64) // 8K: 1040*256

#define VP9_DBLK_LEFT_SIZE_IN_SB64          ALIGN_UP(1600, 128) // 420
#define VP9_DBLK_LEFT_BUF_SIZE(h)           NVDEC_ALIGN( (ALIGN_UP(h,64)/64 + 2) * VP9_DBLK_LEFT_SIZE_IN_SB64) // 8K: 845*256

// VP9 Hint Dump Buffer
#define VP9_HINT_DUMP_SIZE_IN_SB64          ((64*64)/(4*4)*8)           // 8 bytes per CU, 256 CUs(2048 bytes) per SB64
#define VP9_HINT_DUMP_SIZE(w, h)            NVDEC_ALIGN(VP9_HINT_DUMP_SIZE_IN_SB64*((w+63)/64)*((h+63)/64))

// used for ecdma debug
typedef struct _nvdec_ecdma_config_s
{
    unsigned int            ecdma_enable;                               // enable/disable  ecdma
    unsigned short          ecdma_blk_x_src;                            // src start position x , it's 64x aligned
    unsigned short          ecdma_blk_y_src;                            // src start position y , it's 8x aligned
    unsigned short          ecdma_blk_x_dst;                            // dst start position x , it's 64x aligned
    unsigned short          ecdma_blk_y_dst;                            // dst start position y , it's 8x aligned
    unsigned short          ref_pic_idx;                                // ref(src) picture index , used to derived source picture base address
    unsigned short          boundary0_top;                              // src insided tile/partition region top boundary
    unsigned short          boundary0_bottom;                           // src insided tile/partition region bottom boundary
    unsigned short          boundary1_left;                             // src insided tile/partition region left boundary
    unsigned short          boundary1_right;                            // src insided tile/partition region right boundary
    unsigned char           blk_copy_flag;                              // blk_copy enable flag.
                                                                        // if it's 1 ,ctb_size ==3,ecdma_blk_x_src == boundary1_left and ecdma_blk_y_src == boundary0_top ;
                                                                        // if it's 0 ,ecdma_blk_x_src == ecdma_blk_x_dst and ecdma_blk_y_src == ecdma_blk_y_dst;
    unsigned char           ctb_size;                                   // ctb_size .0:64x64,1:32x32,2:16x16,3:8x8
} nvdec_ecdma_config_s;

typedef struct _nvdec_status_hevc_s
{
    unsigned int frame_status_intra_cnt;    //Intra block counter, in unit of 8x8 block, IPCM block included
    unsigned int frame_status_inter_cnt;    //Inter block counter, in unit of 8x8 block, SKIP block included
    unsigned int frame_status_skip_cnt;     //Skip block counter, in unit of 4x4 block, blocks having NO/ZERO texture/coeff data
    unsigned int frame_status_fwd_mvx_cnt;  //ABS sum of forward  MVx, one 14bit MVx(integer) per 4x4 block
    unsigned int frame_status_fwd_mvy_cnt;  //ABS sum of forward  MVy, one 14bit MVy(integer) per 4x4 block
    unsigned int frame_status_bwd_mvx_cnt;  //ABS sum of backward MVx, one 14bit MVx(integer) per 4x4 block
    unsigned int frame_status_bwd_mvy_cnt;  //ABS sum of backward MVy, one 14bit MVy(integer) per 4x4 block
    unsigned int error_ctb_pos;             //[15:0] error ctb   position in Y direction, [31:16] error ctb   position in X direction
    unsigned int error_slice_pos;           //[15:0] error slice position in Y direction, [31:16] error slice position in X direction
} nvdec_status_hevc_s;

typedef struct _nvdec_status_vp9_s
{
    unsigned int frame_status_intra_cnt;    //Intra block counter, in unit of 8x8 block, IPCM block included
    unsigned int frame_status_inter_cnt;    //Inter block counter, in unit of 8x8 block, SKIP block included
    unsigned int frame_status_skip_cnt;     //Skip block counter, in unit of 4x4 block, blocks having NO/ZERO texture/coeff data
    unsigned int frame_status_fwd_mvx_cnt;  //ABS sum of forward  MVx, one 14bit MVx(integer) per 4x4 block
    unsigned int frame_status_fwd_mvy_cnt;  //ABS sum of forward  MVy, one 14bit MVy(integer) per 4x4 block
    unsigned int frame_status_bwd_mvx_cnt;  //ABS sum of backward MVx, one 14bit MVx(integer) per 4x4 block
    unsigned int frame_status_bwd_mvy_cnt;  //ABS sum of backward MVy, one 14bit MVy(integer) per 4x4 block
    unsigned int error_ctb_pos;             //[15:0] error ctb   position in Y direction, [31:16] error ctb   position in X direction
    unsigned int error_slice_pos;           //[15:0] error slice position in Y direction, [31:16] error slice position in X direction
} nvdec_status_vp9_s;

typedef struct _nvdec_status_s
{
    unsigned int    mbs_correctly_decoded;          // total numers of correctly decoded macroblocks
    unsigned int    mbs_in_error;                   // number of error macroblocks.
    unsigned int    cycle_count;                    // total cycles taken for execute. read from PERF_DECODE_FRAME_V register
    unsigned int    error_status;                   // report error if any
    union
    {
        nvdec_status_hevc_s hevc;
        nvdec_status_vp9_s vp9;
    };
    unsigned int    slice_header_error_code;        // report error in slice header

} nvdec_status_s;

// per 16x16 block, used in hevc/vp9 surface of SetExternalMVBufferOffset when error_external_mv_en = 1
typedef struct _external_mv_s
{
    int             mvx     : 14;   //integrate pixel precision
    int             mvy     : 14;   //integrate pixel precision
    unsigned int    refidx  :  4;
} external_mv_s;

// HEVC
typedef struct _nvdec_hevc_main10_444_ext_s
{
    unsigned int transformSkipRotationEnableFlag : 1;    //sps extension for transform_skip_rotation_enabled_flag
    unsigned int transformSkipContextEnableFlag : 1;     //sps extension for transform_skip_context_enabled_flag
    unsigned int intraBlockCopyEnableFlag :1;            //sps intraBlockCopyEnableFlag, always 0 before spec define it
    unsigned int implicitRdpcmEnableFlag : 1;            //sps implicit_rdpcm_enabled_flag
    unsigned int explicitRdpcmEnableFlag : 1;            //sps explicit_rdpcm_enabled_flag
    unsigned int extendedPrecisionProcessingFlag : 1;    //sps extended_precision_processing_flag,always 0 in current profile
    unsigned int intraSmoothingDisabledFlag : 1;         //sps intra_smoothing_disabled_flag
    unsigned int highPrecisionOffsetsEnableFlag :1;      //sps high_precision_offsets_enabled_flag
    unsigned int fastRiceAdaptationEnableFlag: 1;        //sps fast_rice_adaptation_enabled_flag
    unsigned int cabacBypassAlignmentEnableFlag : 1;     //sps cabac_bypass_alignment_enabled_flag, always 0 in current profile
    unsigned int sps_444_extension_reserved : 22;        //sps reserve for future extension

    unsigned int log2MaxTransformSkipSize : 4 ;          //pps extension log2_max_transform_skip_block_size_minus2, 0...5
    unsigned int crossComponentPredictionEnableFlag: 1;  //pps cross_component_prediction_enabled_flag
    unsigned int chromaQpAdjustmentEnableFlag:1;         //pps chroma_qp_adjustment_enabled_flag
    unsigned int diffCuChromaQpAdjustmentDepth:2;        //pps diff_cu_chroma_qp_adjustment_depth, 0...3
    unsigned int chromaQpAdjustmentTableSize:3;          //pps chroma_qp_adjustment_table_size_minus1+1, 1...6
    unsigned int log2SaoOffsetScaleLuma:3;               //pps log2_sao_offset_scale_luma, max(0,bitdepth-10),maxBitdepth 16 for future.
    unsigned int log2SaoOffsetScaleChroma: 3;            //pps log2_sao_offset_scale_chroma
    unsigned int pps_444_extension_reserved : 15;        //pps reserved
    char         cb_qp_adjustment[6];                    //-[12,+12]
    char         cr_qp_adjustment[6];                    //-[12,+12]
    unsigned int   HevcFltAboveOffset;  // filter above offset respect to filter buffer, 256 bytes unit
    unsigned int   HevcSaoAboveOffset;  // sao    above offset respect to filter buffer, 256 bytes unit
} nvdec_hevc_main10_444_ext_s;

typedef struct _nvdec_hevc_pic_v1_s
{
    // New fields
    //hevc main10 444 extensions
    nvdec_hevc_main10_444_ext_s hevc_main10_444_ext;

    //HEVC skip bytes from beginning setting for secure
    //it is different to the sw_hdr_skip_length who skips the middle of stream of
    //the slice header which is parsed by driver
    unsigned int   sw_skip_start_length : 14;
    unsigned int   external_ref_mem_dis :  1;
    unsigned int   error_recovery_start_pos :  2;       //0: from start of frame, 1: from start of slice segment, 2: from error detected ctb, 3: reserved
    unsigned int   error_external_mv_en :  1;
    unsigned int   reserved0            : 14;
    // Reserved bits padding
} nvdec_hevc_pic_v1_s;

//No versioning in structure: NVDEC2 (T210 and GM206)
//version v1 : NVDEC3 (T186 and GP100)
//version v2 : NVDEC3.1 (GP10x)

typedef struct _nvdec_hevc_pic_v2_s
{
    // mv-hevc field
    unsigned  int  mv_hevc_enable                     :1;
    unsigned  int  nuh_layer_id                       :6;
    unsigned  int  default_ref_layers_active_flag     :1;
    unsigned  int  NumDirectRefLayers                 :6;
    unsigned  int  max_one_active_ref_layer_flag      :1;
    unsigned  int  NumActiveRefLayerPics              :6;
    unsigned  int  poc_lsb_not_present_flag           :1;
    unsigned  int  reserved0                          :10;
} nvdec_hevc_pic_v2_s;

typedef struct _nvdec_hevc_pic_v3_s
{
    // slice level decoding
    unsigned  int  slice_decoding_enable:1;//1: enable slice level decoding
    unsigned  int  slice_ec_enable:1;      //1: enable slice error concealment. When slice_ec_enable=1,slice_decoding_enable must be 1;
    unsigned  int  slice_ec_mv_type:2;     //0: zero mv; 1: co-located mv; 2: external mv;
    unsigned  int  err_detected_sw:1;      //1: indicate sw/driver has detected error already in frame kick mode
    unsigned  int  slice_ec_slice_type:2;  //0: B slice; 1: P slice ; others: reserved
    unsigned  int  slice_strm_recfg_en:1;  //enable slice bitstream re-configure or not ;
    unsigned  int  reserved:24;
    unsigned  int  HevcSliceEdgeOffset;// slice edge buffer offset which repsect to filter buffer ,256 bytes as one unit
}nvdec_hevc_pic_v3_s;

typedef struct _nvdec_hevc_pic_s
{
    //The key/IV addr must be 128bit alignment
    unsigned int   wrapped_session_key[4];                      //session keys
    unsigned int   wrapped_content_key[4];                      //content keys
    unsigned int   initialization_vector[4];                    //Ctrl64 initial vector
    // hevc_bitstream_data_info
    unsigned int   stream_len;                                  // stream length in one frame
    unsigned int   enable_encryption;                           // flag to enable/disable encryption
    unsigned int   key_increment   : 6;                           // added to content key after unwrapping
    unsigned int   encryption_mode : 4;
    unsigned int   key_slot_index  : 4;
    unsigned int   ssm_en          : 1;
    unsigned int   enable_histogram  : 1;                       // histogram stats output enable
    unsigned int   enable_substream_decoding: 1;            //frame substream kickoff without context switch
    unsigned int   reserved0       :15;

    // Driver may or may not use based upon need.
    // If 0 then default value of 1<<27 = 298ms @ 450MHz will be used in ucode.
    // Driver can send this value based upon resolution using the formula:
    // gptimer_timeout_value = 3 * (cycles required for one frame)
    unsigned int gptimer_timeout_value;

    // general
    unsigned char tileformat                 : 2 ;   // 0: TBL; 1: KBL; 2: Tile16x16
    unsigned char gob_height                 : 3 ;   // Set GOB height, 0: GOB_2, 1: GOB_4, 2: GOB_8, 3: GOB_16, 4: GOB_32 (NVDEC3 onwards)
    unsigned char reserverd_surface_format   : 3 ;
    unsigned char sw_start_code_e;                             // 0: stream doesn't contain start codes,1: stream contains start codes
    unsigned char disp_output_mode;                            // 0: Rec.709 8 bit, 1: Rec.709 10 bit, 2: Rec.709 10 bits -> 8 bit, 3: Rec.2020 10 bit -> 8 bit
    unsigned char reserved1;
    unsigned int  framestride[2];                              // frame buffer stride for luma and chroma
    unsigned int  colMvBuffersize;                             // collocated MV buffer size of one picture ,256 bytes unit
    unsigned int  HevcSaoBufferOffset;                         // sao buffer offset respect to filter buffer ,256 bytes unit .
    unsigned int  HevcBsdCtrlOffset;                           // bsd buffer offset respect to filter buffer ,256 bytes unit .
    // sps
    unsigned short pic_width_in_luma_samples;                      // :15, 48(?)..16384, multiple of 8 (48 is smallest width supported by NVDEC for CTU size 16x16)
    unsigned short pic_height_in_luma_samples;                     // :15, 8..16384, multiple of 8
    unsigned int chroma_format_idc                            : 4; // always 1 (=4:2:0)
    unsigned int bit_depth_luma                               : 4; // 8..12
    unsigned int bit_depth_chroma                             : 4;
    unsigned int log2_min_luma_coding_block_size              : 4; // 3..6
    unsigned int log2_max_luma_coding_block_size              : 4; // 3..6
    unsigned int log2_min_transform_block_size                : 4; // 2..5
    unsigned int log2_max_transform_block_size                : 4; // 2..5
    unsigned int reserved2                                    : 4;

    unsigned int max_transform_hierarchy_depth_inter          : 3; // 0..4
    unsigned int max_transform_hierarchy_depth_intra          : 3; // 0..4
    unsigned int scalingListEnable                            : 1; //
    unsigned int amp_enable_flag                              : 1; //
    unsigned int sample_adaptive_offset_enabled_flag          : 1; //
    unsigned int pcm_enabled_flag                             : 1; //
    unsigned int pcm_sample_bit_depth_luma                    : 4; //
    unsigned int pcm_sample_bit_depth_chroma                  : 4;
    unsigned int log2_min_pcm_luma_coding_block_size          : 4; //
    unsigned int log2_max_pcm_luma_coding_block_size          : 4; //
    unsigned int pcm_loop_filter_disabled_flag                : 1; //
    unsigned int sps_temporal_mvp_enabled_flag                : 1; //
    unsigned int strong_intra_smoothing_enabled_flag          : 1; //
    unsigned int reserved3                                    : 3;
    // pps
    unsigned int dependent_slice_segments_enabled_flag        : 1; //
    unsigned int output_flag_present_flag                     : 1; //
    unsigned int num_extra_slice_header_bits                  : 3; //  0..7 (normally 0)
    unsigned int sign_data_hiding_enabled_flag                : 1; //
    unsigned int cabac_init_present_flag                      : 1; //
    unsigned int num_ref_idx_l0_default_active                : 4; //  1..15
    unsigned int num_ref_idx_l1_default_active                : 4; //  1..15
    unsigned int init_qp                                      : 7; //  0..127, support higher bitdepth
    unsigned int constrained_intra_pred_flag                  : 1; //
    unsigned int transform_skip_enabled_flag                  : 1; //
    unsigned int cu_qp_delta_enabled_flag                     : 1; //
    unsigned int diff_cu_qp_delta_depth                       : 2; //  0..3
    unsigned int reserved4                                    : 5; //

    char         pps_cb_qp_offset                             ; //  -12..12
    char         pps_cr_qp_offset                             ; //  -12..12
    char         pps_beta_offset                              ; //  -12..12
    char         pps_tc_offset                                ; //  -12..12
    unsigned int pps_slice_chroma_qp_offsets_present_flag     : 1; //
    unsigned int weighted_pred_flag                           : 1; //
    unsigned int weighted_bipred_flag                         : 1; //
    unsigned int transquant_bypass_enabled_flag               : 1; //
    unsigned int tiles_enabled_flag                           : 1; // (redundant: = num_tile_columns_minus1!=0 || num_tile_rows_minus1!=0)
    unsigned int entropy_coding_sync_enabled_flag             : 1; //
    unsigned int num_tile_columns                             : 5; // 0..20
    unsigned int num_tile_rows                                : 5; // 0..22
    unsigned int loop_filter_across_tiles_enabled_flag        : 1; //
    unsigned int loop_filter_across_slices_enabled_flag       : 1; //
    unsigned int deblocking_filter_control_present_flag       : 1; //
    unsigned int deblocking_filter_override_enabled_flag      : 1; //
    unsigned int pps_deblocking_filter_disabled_flag          : 1; //
    unsigned int lists_modification_present_flag              : 1; //
    unsigned int log2_parallel_merge_level                    : 3; //  2..4
    unsigned int slice_segment_header_extension_present_flag  : 1; // (normally 0)
    unsigned int reserved5                                    : 6;

    // reference picture related
    unsigned char  num_ref_frames;
    unsigned char  reserved6;
    unsigned short longtermflag;                              // long term flag for refpiclist.bit 15 for picidx 0, bit 14 for picidx 1,...
    unsigned char  initreflistidxl0[16];                           // :5, [refPicidx] 0..15
    unsigned char  initreflistidxl1[16];                           // :5, [refPicidx] 0..15
    short          RefDiffPicOrderCnts[16];                     // poc diff between current and reference pictures .[-128,127]
    // misc
    unsigned char  IDR_picture_flag;                            // idr flag for current picture
    unsigned char  RAP_picture_flag;                            // rap flag for current picture
    unsigned char  curr_pic_idx;                                // current  picture store buffer index,used to derive the store addess of frame buffer and MV
    unsigned char  pattern_id;                                  // used for dithering to select between 2 tables
    unsigned short sw_hdr_skip_length;                          // reference picture inititial related syntax elements(SE) bits in slice header.
                                                                // those SE only decoding once in driver,related bits will flush in HW
    unsigned short reserved7;

    // used for ecdma debug
    nvdec_ecdma_config_s  ecdma_cfg;

    //DXVA on windows
    unsigned int   separate_colour_plane_flag : 1;
    unsigned int   log2_max_pic_order_cnt_lsb_minus4 : 4;    //0~12
    unsigned int   num_short_term_ref_pic_sets : 7 ;  //0~64
    unsigned int   num_long_term_ref_pics_sps :  6;  //0~32
    unsigned int   bBitParsingDisable : 1 ; //disable parsing
    unsigned int   num_delta_pocs_of_rps_idx : 8;
    unsigned int   long_term_ref_pics_present_flag : 1;
    unsigned int   reserved_dxva : 4;
    //the number of bits for short_term_ref_pic_set()in slice header,dxva API
    unsigned int   num_bits_short_term_ref_pics_in_slice;

    // New additions
    nvdec_hevc_pic_v1_s v1;
    nvdec_hevc_pic_v2_s v2;
    nvdec_hevc_pic_v3_s v3;
    nvdec_pass2_otf_ext_s ssm;

} nvdec_hevc_pic_s;

//hevc slice info class
typedef struct _hevc_slice_info_s {
    unsigned int   first_flag    :1;//first slice(s) of frame,must valid for slice EC
    unsigned int   err_flag      :1;//error slice(s) .optional info for EC
    unsigned int   last_flag     :1;//last slice segment(s) of frame,this bit is must be valid when slice_strm_recfg_en==1 or slice_ec==1
    unsigned int   conceal_partial_slice :1; // indicate do partial slice error conealment for packet loss case
    unsigned int   available     :1; // indicate the slice bitstream is available.
    unsigned int   reserved0     :7;
    unsigned int   ctb_count     :20;// ctbs counter inside slice(s) .must valid for slice EC
    unsigned int   bs_offset; //slice(s) bitstream offset in bitstream buffer (in byte unit)
    unsigned int   bs_length; //slice(s) bitstream length. It is sum of aligned size and skip size and valid slice bitstream size.
    unsigned short start_ctbx; //slice start ctbx ,it's optional,HW can output it in previous slice decoding.
                                //but this is one check points for error
    unsigned short start_ctby; //slice start ctby
 } hevc_slice_info_s;


//hevc slice ctx class
//slice pos and next slice address
typedef struct  _slice_edge_ctb_pos_ctx_s {
    unsigned int    next_slice_pos_ctbxy;         //2d address in raster scan
    unsigned int    next_slice_segment_addr;      //1d address in  tile scan
}slice_edge_ctb_pos_ctx_s;

//  next slice's first ctb located tile related information
typedef struct  _slice_edge_tile_ctx_s {
    unsigned int    tileInfo1;// Misc tile info includes tile width and tile height and tile col and tile row
    unsigned int    tileInfo2;// Misc tile info includes tile start ctbx and start ctby and tile index
    unsigned int    tileInfo3;// Misc tile info includes  ctb pos inside tile
} slice_edge_tile_ctx_s;

//frame level stats
typedef struct  _slice_edge_stats_ctx_s {
    unsigned int    frame_status_intra_cnt;// frame stats for intra block count
    unsigned int    frame_status_inter_cnt;// frame stats for inter block count
    unsigned int    frame_status_skip_cnt;// frame stats for skip block count
    unsigned int    frame_status_fwd_mvx_cnt;// frame stats for sum of  abs fwd mvx
    unsigned int    frame_status_fwd_mvy_cnt;// frame stats for sum of  abs fwd mvy
    unsigned int    frame_status_bwd_mvx_cnt;// frame stats for sum of  abs bwd mvx
    unsigned int    frame_status_bwd_mvy_cnt;// frame stats for sum of  abs bwd mvy
    unsigned int    frame_status_mv_cnt_ext;// extension bits of  sum of abs mv to keep full precision.
}slice_edge_stats_ctx_s;

//ctx of vpc_edge unit for tile left
typedef struct  _slice_vpc_edge_ctx_s {
    unsigned int   reserved;
}slice_vpc_edge_ctx_s;

//ctx of vpc_main unit
typedef struct  _slice_vpc_main_ctx_s {
    unsigned int   reserved;
} slice_vpc_main_ctx_s;

//hevc slice edge ctx class
typedef struct  _slice_edge_ctx_s {
    //ctb pos
    slice_edge_ctb_pos_ctx_s  slice_ctb_pos_ctx;
    // stats
    slice_edge_stats_ctx_s slice_stats_ctx;
    // tile info
    slice_edge_tile_ctx_s    slice_tile_ctx;
    //vpc_edge
    slice_vpc_edge_ctx_s  slice_vpc_edge_ctx;
    //vpc_main
    slice_vpc_main_ctx_s  slice_vpc_main_ctx;
} slice_edge_ctx_s;

//vp9

typedef struct _nvdec_vp9_pic_v1_s
{
    // New fields
    // new_var : xx; // for variables with expanded bitlength, comment on why the new bit legth is required
    // Reserved bits for padding and/or non-HW specific functionality
    unsigned int   Vp9FltAboveOffset;  // filter above offset respect to filter buffer, 256 bytes unit
    unsigned int   external_ref_mem_dis :  1;
    unsigned int   bit_depth            :  4;
    unsigned int   error_recovery_start_pos :  2;       //0: from start of frame, 1: from start of slice segment, 2: from error detected ctb, 3: reserved
    unsigned int   error_external_mv_en :  1;
    unsigned int   Reserved0            : 24;
} nvdec_vp9_pic_v1_s;

enum VP9_FRAME_SFC_ID
{
    VP9_LAST_FRAME_SFC = 0,
    VP9_GOLDEN_FRAME_SFC,
    VP9_ALTREF_FRAME_SFC,
    VP9_CURR_FRAME_SFC
};

typedef struct _nvdec_vp9_pic_s
{
    // vp9_bitstream_data_info
    //Key and IV address must 128bit alignment
    unsigned int   wrapped_session_key[4];                      //session keys
    unsigned int   wrapped_content_key[4];                      //content keys
    unsigned int   initialization_vector[4];                    //Ctrl64 initial vector
    unsigned int   stream_len;                                  // stream length in one frame
    unsigned int   enable_encryption;                           // flag to enable/disable encryption
    unsigned int   key_increment      : 6;                      // added to content key after unwrapping
    unsigned int   encryption_mode    : 4;
    unsigned int   sw_hdr_skip_length :14;                      //vp9 skip bytes setting for secure
    unsigned int   key_slot_index     : 4;
    unsigned int   ssm_en             : 1;
    unsigned int   enable_histogram   : 1;                      // histogram stats output enable
    unsigned int   reserved0          : 2;

    // Driver may or may not use based upon need.
    // If 0 then default value of 1<<27 = 298ms @ 450MHz will be used in ucode.
    // Driver can send this value based upon resolution using the formula:
    // gptimer_timeout_value = 3 * (cycles required for one frame)
    unsigned int gptimer_timeout_value;

    //general
    unsigned char  tileformat                 : 2 ;   // 0: TBL; 1: KBL; 2: Tile16x16
    unsigned char  gob_height                 : 3 ;   // Set GOB height, 0: GOB_2, 1: GOB_4, 2: GOB_8, 3: GOB_16, 4: GOB_32 (NVDEC3 onwards)
    unsigned char  reserverd_surface_format   : 3 ;
    unsigned char  reserved1[3];
    unsigned int   Vp9BsdCtrlOffset;                           // bsd buffer offset respect to filter buffer ,256 bytes unit .


    //ref_last dimensions
    unsigned short  ref0_width;    //ref_last coded width
    unsigned short  ref0_height;   //ref_last coded height
    unsigned short  ref0_stride[2];    //ref_last stride

    //ref_golden dimensions
    unsigned short  ref1_width;    //ref_golden coded width
    unsigned short  ref1_height;   //ref_golden coded height
    unsigned short  ref1_stride[2];    //ref_golden stride

    //ref_alt dimensions
    unsigned short  ref2_width;    //ref_alt coded width
    unsigned short  ref2_height;   //ref_alt coded height
    unsigned short  ref2_stride[2];    //ref_alt stride


    /* Current frame dimensions */
    unsigned short  width;    //pic width
    unsigned short  height;   //pic height
    unsigned short  framestride[2];   // frame buffer stride for luma and chroma

    unsigned char   keyFrame  :1;
    unsigned char   prevIsKeyFrame:1;
    unsigned char   resolutionChange:1;
    unsigned char   errorResilient:1;
    unsigned char   prevShowFrame:1;
    unsigned char   intraOnly:1;
    unsigned char   reserved2 : 2;

    /* DCT coefficient partitions */
    //unsigned int    offsetToDctParts;

    unsigned char   reserved3[3];
    //unsigned char   activeRefIdx[3];//3 bits
    //unsigned char   refreshFrameFlags;
    //unsigned char   refreshEntropyProbs;
    //unsigned char   frameParallelDecoding;
    //unsigned char   resetFrameContext;

    unsigned char   refFrameSignBias[4];
    char            loopFilterLevel;//6 bits
    char            loopFilterSharpness;//3 bits

    /* Quantization parameters */
    unsigned char   qpYAc;
    char            qpYDc;
    char            qpChAc;
    char            qpChDc;

    /* From here down, frame-to-frame persisting stuff */

    char            lossless;
    char            transform_mode;
    char            allow_high_precision_mv;
    char            mcomp_filter_type;
    char            comp_pred_mode;
    char            comp_fixed_ref;
    char            comp_var_ref[2];
    char            log2_tile_columns;
    char            log2_tile_rows;

    /* Segment and macroblock specific values */
    unsigned char   segmentEnabled;
    unsigned char   segmentMapUpdate;
    unsigned char   segmentMapTemporalUpdate;
    unsigned char   segmentFeatureMode; /* ABS data or delta data */
    unsigned char   segmentFeatureEnable[8][4];
    short           segmentFeatureData[8][4];
    char            modeRefLfEnabled;
    char            mbRefLfDelta[4];
    char            mbModeLfDelta[2];
    char            reserved5;            // for alignment

    // New additions
    nvdec_vp9_pic_v1_s v1;
    nvdec_pass2_otf_ext_s ssm;

} nvdec_vp9_pic_s;

#define NVDEC_VP9HWPAD(x, y) unsigned char x[y]

typedef struct {
    /* last bytes of address 41 */
    unsigned char joints[3];
    unsigned char sign[2];
    /* address 42 */
    unsigned char class0[2][1];
    unsigned char fp[2][3];
    unsigned char class0_hp[2];
    unsigned char hp[2];
    unsigned char classes[2][10];
    /* address 43 */
    unsigned char class0_fp[2][2][3];
    unsigned char bits[2][10];

} nvdec_nmv_context;

typedef struct {
    unsigned int joints[4];
    unsigned int sign[2][2];
    unsigned int classes[2][11];
    unsigned int class0[2][2];
    unsigned int bits[2][10][2];
    unsigned int class0_fp[2][2][4];
    unsigned int fp[2][4];
    unsigned int class0_hp[2][2];
    unsigned int hp[2][2];

} nvdec_nmv_context_counts;

/* Adaptive entropy contexts, padding elements are added to have
 * 256 bit aligned tables for HW access.
 * Compile with TRACE_PROB_TABLES to print bases for each table. */
typedef struct nvdec_vp9AdaptiveEntropyProbs_s
{
    /* address 32 */
    unsigned char inter_mode_prob[7][4];
    unsigned char intra_inter_prob[4];

    /* address 33 */
    unsigned char uv_mode_prob[10][8];
    unsigned char tx8x8_prob[2][1];
    unsigned char tx16x16_prob[2][2];
    unsigned char tx32x32_prob[2][3];
    unsigned char sb_ymode_probB[4][1];
    unsigned char sb_ymode_prob[4][8];

    /* address 37 */
    unsigned char partition_prob[2][16][4];

    /* address 41 */
    unsigned char uv_mode_probB[10][1];
    unsigned char switchable_interp_prob[4][2];
    unsigned char comp_inter_prob[5];
    unsigned char mbskip_probs[3];
    NVDEC_VP9HWPAD(pad1, 1);

    nvdec_nmv_context nmvc;

    /* address 44 */
    unsigned char single_ref_prob[5][2];
    unsigned char comp_ref_prob[5];
    NVDEC_VP9HWPAD(pad2, 17);

    /* address 45 */
    unsigned char probCoeffs[2][2][6][6][4];
    unsigned char probCoeffs8x8[2][2][6][6][4];
    unsigned char probCoeffs16x16[2][2][6][6][4];
    unsigned char probCoeffs32x32[2][2][6][6][4];

} nvdec_vp9AdaptiveEntropyProbs_t;

/* Entropy contexts */
typedef struct nvdec_vp9EntropyProbs_s
{
    /* Default keyframe probs */
    /* Table formatted for 256b memory, probs 0to7 for all tables followed by
     * probs 8toN for all tables.
     * Compile with TRACE_PROB_TABLES to print bases for each table. */

    unsigned char kf_bmode_prob[10][10][8];

    /* Address 25 */
    unsigned char kf_bmode_probB[10][10][1];
    unsigned char ref_pred_probs[3];
    unsigned char mb_segment_tree_probs[7];
    unsigned char segment_pred_probs[3];
    unsigned char ref_scores[4];
    unsigned char prob_comppred[2];
    NVDEC_VP9HWPAD(pad1, 9);

    /* Address 29 */
    unsigned char kf_uv_mode_prob[10][8];
    unsigned char kf_uv_mode_probB[10][1];
    NVDEC_VP9HWPAD(pad2, 6);

    nvdec_vp9AdaptiveEntropyProbs_t a;    /* Probs with backward adaptation */

} nvdec_vp9EntropyProbs_t;

/* Counters for adaptive entropy contexts */
typedef struct nvdec_vp9EntropyCounts_s
{
    unsigned int inter_mode_counts[7][3][2];
    unsigned int sb_ymode_counts[4][10];
    unsigned int uv_mode_counts[10][10];
    unsigned int partition_counts[16][4];
    unsigned int switchable_interp_counts[4][3];
    unsigned int intra_inter_count[4][2];
    unsigned int comp_inter_count[5][2];
    unsigned int single_ref_count[5][2][2];
    unsigned int comp_ref_count[5][2];
    unsigned int tx32x32_count[2][4];
    unsigned int tx16x16_count[2][3];
    unsigned int tx8x8_count[2][2];
    unsigned int mbskip_count[3][2];

    nvdec_nmv_context_counts nmvcount;

    unsigned int countCoeffs[2][2][6][6][4];
    unsigned int countCoeffs8x8[2][2][6][6][4];
    unsigned int countCoeffs16x16[2][2][6][6][4];
    unsigned int countCoeffs32x32[2][2][6][6][4];

    unsigned int countEobs[4][2][2][6][6];

} nvdec_vp9EntropyCounts_t;

// Legacy codecs encryption parameters
typedef struct _nvdec_pass2_otf_s {
    unsigned int   wrapped_session_key[4];  // session keys
    unsigned int   wrapped_content_key[4];  // content keys
    unsigned int   initialization_vector[4];// Ctrl64 initial vector
    unsigned int   enable_encryption : 1;   // flag to enable/disable encryption
    unsigned int   key_increment     : 6;   // added to content key after unwrapping
    unsigned int   encryption_mode   : 4;
    unsigned int   key_slot_index    : 4;
    unsigned int   ssm_en            : 1;
    unsigned int   reserved1         :16;   // reserved
} nvdec_pass2_otf_s; // 0x10 bytes

typedef struct _nvdec_display_param_s
{
    unsigned int enableTFOutput    : 1; //=1, enable dbfdma to output the display surface; if disable, then the following configure on tf is useless.
    //remap for VC1
    unsigned int VC1MapYFlag       : 1;
    unsigned int MapYValue         : 3;
    unsigned int VC1MapUVFlag      : 1;
    unsigned int MapUVValue        : 3;
    //tf
    unsigned int OutStride         : 8;
    unsigned int TilingFormat      : 3;
    unsigned int OutputStructure   : 1; //(0=frame, 1=field)
    unsigned int reserved0         :11;
    int OutputTop[2];                   // in units of 256
    int OutputBottom[2];                // in units of 256
    //histogram
    unsigned int enableHistogram   : 1; // enable histogram info collection.
    unsigned int HistogramStartX   :12; // start X of Histogram window
    unsigned int HistogramStartY   :12; // start Y of Histogram window
    unsigned int reserved1         : 7;
    unsigned int HistogramEndX     :12; // end X of Histogram window
    unsigned int HistogramEndY     :12; // end y of Histogram window
    unsigned int reserved2         : 8;
} nvdec_display_param_s;  // size 0x1c bytes

// H.264
typedef struct _nvdec_dpb_entry_s  // 16 bytes
{
    unsigned int index          : 7;    // uncompressed frame buffer index
    unsigned int col_idx        : 5;    // index of associated co-located motion data buffer
    unsigned int state          : 2;    // bit1(state)=1: top field used for reference, bit1(state)=1: bottom field used for reference
    unsigned int is_long_term   : 1;    // 0=short-term, 1=long-term
    unsigned int not_existing   : 1;    // 1=marked as non-existing
    unsigned int is_field       : 1;    // set if unpaired field or complementary field pair
    unsigned int top_field_marking : 4;
    unsigned int bottom_field_marking : 4;
    unsigned int output_memory_layout : 1;  // Set according to picture level output NV12/NV24 setting.
    unsigned int reserved       : 6;
    unsigned int FieldOrderCnt[2];      // : 2*32 [top/bottom]
    int FrameIdx;                       // : 16   short-term: FrameNum (16 bits), long-term: LongTermFrameIdx (4 bits)
} nvdec_dpb_entry_s;

typedef struct _nvdec_h264_pic_s
{
    nvdec_pass2_otf_s encryption_params;
    unsigned char eos[16];
    unsigned char explicitEOSPresentFlag;
    unsigned char hint_dump_en; //enable COLOMV surface dump for all frames, which includes hints of "MV/REFIDX/QP/CBP/MBPART/MBTYPE", nvbug: 200212874
    unsigned char reserved0[2];
    unsigned int stream_len;
    unsigned int slice_count;
    unsigned int mbhist_buffer_size;     // to pass buffer size of MBHIST_BUFFER

    // Driver may or may not use based upon need.
    // If 0 then default value of 1<<27 = 298ms @ 450MHz will be used in ucode.
    // Driver can send this value based upon resolution using the formula:
    // gptimer_timeout_value = 3 * (cycles required for one frame)
    unsigned int gptimer_timeout_value;

    // Fields from msvld_h264_seq_s
    int log2_max_pic_order_cnt_lsb_minus4;
    int delta_pic_order_always_zero_flag;
    int frame_mbs_only_flag;
    int PicWidthInMbs;
    int FrameHeightInMbs;

    unsigned int tileFormat                 : 2 ;   // 0: TBL; 1: KBL; 2: Tile16x16
    unsigned int gob_height                 : 3 ;   // Set GOB height, 0: GOB_2, 1: GOB_4, 2: GOB_8, 3: GOB_16, 4: GOB_32 (NVDEC3 onwards)
    unsigned int reserverd_surface_format   : 27;

    // Fields from msvld_h264_pic_s
    int entropy_coding_mode_flag;
    int pic_order_present_flag;
    int num_ref_idx_l0_active_minus1;
    int num_ref_idx_l1_active_minus1;
    int deblocking_filter_control_present_flag;
    int redundant_pic_cnt_present_flag;
    int transform_8x8_mode_flag;

    // Fields from mspdec_h264_picture_setup_s
    unsigned int pitch_luma;                    // Luma pitch
    unsigned int pitch_chroma;                  // chroma pitch

    unsigned int luma_top_offset;               // offset of luma top field in units of 256
    unsigned int luma_bot_offset;               // offset of luma bottom field in units of 256
    unsigned int luma_frame_offset;             // offset of luma frame in units of 256
    unsigned int chroma_top_offset;             // offset of chroma top field in units of 256
    unsigned int chroma_bot_offset;             // offset of chroma bottom field in units of 256
    unsigned int chroma_frame_offset;           // offset of chroma frame in units of 256
    unsigned int HistBufferSize;                // in units of 256

    unsigned int MbaffFrameFlag           : 1;  //
    unsigned int direct_8x8_inference_flag: 1;  //
    unsigned int weighted_pred_flag       : 1;  //
    unsigned int constrained_intra_pred_flag:1; //
    unsigned int ref_pic_flag             : 1;  // reference picture (nal_ref_idc != 0)
    unsigned int field_pic_flag           : 1;  //
    unsigned int bottom_field_flag        : 1;  //
    unsigned int second_field             : 1;  // second field of complementary reference field
    unsigned int log2_max_frame_num_minus4: 4;  //  (0..12)
    unsigned int chroma_format_idc        : 2;  //
    unsigned int pic_order_cnt_type       : 2;  //  (0..2)
    int pic_init_qp_minus26               : 6;  // : 6 (-26..+25)
    int chroma_qp_index_offset            : 5;  // : 5 (-12..+12)
    int second_chroma_qp_index_offset     : 5;  // : 5 (-12..+12)

    unsigned int weighted_bipred_idc      : 2;  // : 2 (0..2)
    unsigned int CurrPicIdx               : 7;  // : 7  uncompressed frame buffer index
    unsigned int CurrColIdx               : 5;  // : 5  index of associated co-located motion data buffer
    unsigned int frame_num                : 16; //
    unsigned int frame_surfaces           : 1;  // frame surfaces flag
    unsigned int output_memory_layout     : 1;  // 0: NV12; 1:NV24. Field pair must use the same setting.

    int CurrFieldOrderCnt[2];                   // : 32 [Top_Bottom], [0]=TopFieldOrderCnt, [1]=BottomFieldOrderCnt
    nvdec_dpb_entry_s dpb[16];
    unsigned char WeightScale[6][4][4];         // : 6*4*4*8 in raster scan order (not zig-zag order)
    unsigned char WeightScale8x8[2][8][8];      // : 2*8*8*8 in raster scan order (not zig-zag order)

    // mvc setup info, must be zero if not mvc
    unsigned char num_inter_view_refs_lX[2];         // number of inter-view references
    char reserved1[14];                               // reserved for alignment
    signed char inter_view_refidx_lX[2][16];         // DPB indices (must also be marked as long-term)

    // lossless decode (At the time of writing this manual, x264 and JM encoders, differ in Intra_8x8 reference sample filtering)
    unsigned int lossless_ipred8x8_filter_enable        : 1;       // = 0, skips Intra_8x8 reference sample filtering, for vertical and horizontal predictions (x264 encoded streams); = 1, filter Intra_8x8 reference samples (JM encoded streams)
    unsigned int qpprime_y_zero_transform_bypass_flag   : 1;       // determines the transform bypass mode
    unsigned int reserved2                              : 30;      // kept for alignment; may be used for other parameters

    nvdec_display_param_s displayPara;
    nvdec_pass2_otf_ext_s ssm;

} nvdec_h264_pic_s;

// VC-1 Scratch buffer
typedef enum _vc1_fcm_e
{
    FCM_PROGRESSIVE = 0,
    FCM_FRAME_INTERLACE = 2,
    FCM_FIELD_INTERLACE = 3
} vc1_fcm_e;

typedef enum _syntax_vc1_ptype_e
{
    PTYPE_I       = 0,
    PTYPE_P       = 1,
    PTYPE_B       = 2,
    PTYPE_BI      = 3, //PTYPE_BI is not used to config register NV_CNVDEC_VLD_PIC_INFO_COMMON. field NV_CNVDEC_VLD_PIC_INFO_COMMON_PIC_CODING_VC1 is only 2 bits. I and BI pictures are configured with same value. Please refer to manual.
    PTYPE_SKIPPED = 4
} syntax_vc1_ptype_e;

// 7.1.1.32, Table 46 etc.
enum vc1_mvmode_e
{
    MVMODE_MIXEDMV                = 0,
    MVMODE_1MV                    = 1,
    MVMODE_1MV_HALFPEL            = 2,
    MVMODE_1MV_HALFPEL_BILINEAR   = 3,
    MVMODE_INTENSITY_COMPENSATION = 4
};

// 9.1.1.42, Table 105
typedef enum _vc1_fptype_e
{
    FPTYPE_I_I = 0,
    FPTYPE_I_P,
    FPTYPE_P_I,
    FPTYPE_P_P,
    FPTYPE_B_B,
    FPTYPE_B_BI,
    FPTYPE_BI_B,
    FPTYPE_BI_BI
} vc1_fptype_e;

// Table 43 (7.1.1.31.2)
typedef enum _vc1_dqprofile_e
{
    DQPROFILE_ALL_FOUR_EDGES  = 0,
    DQPROFILE_DOUBLE_EDGE     = 1,
    DQPROFILE_SINGLE_EDGE     = 2,
    DQPROFILE_ALL_MACROBLOCKS = 3
} vc1_dqprofile_e;

typedef struct _nvdec_vc1_pic_s
{
    nvdec_pass2_otf_s encryption_params;
    unsigned char eos[16];                    // to pass end of stream data separately if not present in bitstream surface
    unsigned char prefixStartCode[4];         // used for dxva to pass prefix start code.
    unsigned int  bitstream_offset;           // offset in words from start of bitstream surface if there is gap.
    unsigned char explicitEOSPresentFlag;     // to indicate that eos[] is used for passing end of stream data.
    unsigned char reserved0[3];
    unsigned int stream_len;
    unsigned int slice_count;
    unsigned int scratch_pic_buffer_size;

    // Driver may or may not use based upon need.
    // If 0 then default value of 1<<27 = 298ms @ 450MHz will be used in ucode.
    // Driver can send this value based upon resolution using the formula:
    // gptimer_timeout_value = 3 * (cycles required for one frame)
    unsigned int gptimer_timeout_value;

    // Fields from vc1_seq_s
    unsigned short FrameWidth;     // actual frame width
    unsigned short FrameHeight;    // actual frame height

    unsigned char profile;        // 1 = SIMPLE or MAIN, 2 = ADVANCED
    unsigned char postprocflag;
    unsigned char pulldown;
    unsigned char interlace;

    unsigned char tfcntrflag;
    unsigned char finterpflag;
    unsigned char psf;
    unsigned char tileFormat                 : 2 ;   // 0: TBL; 1: KBL; 2: Tile16x16
    unsigned char gob_height                 : 3 ;   // Set GOB height, 0: GOB_2, 1: GOB_4, 2: GOB_8, 3: GOB_16, 4: GOB_32 (NVDEC3 onwards)
    unsigned char reserverd_surface_format   : 3 ;

    // simple,main
    unsigned char multires;
    unsigned char syncmarker;
    unsigned char rangered;
    unsigned char maxbframes;

    // Fields from vc1_entrypoint_s
    unsigned char dquant;
    unsigned char panscan_flag;
    unsigned char refdist_flag;
    unsigned char quantizer;

    unsigned char extended_mv;
    unsigned char extended_dmv;
    unsigned char overlap;
    unsigned char vstransform;

    // Fields from vc1_scratch_s
    char refdist;
    char reserved1[3];               // for alignment

    // Fields from vld_vc1_pic_s
    vc1_fcm_e fcm;
    syntax_vc1_ptype_e ptype;
    int tfcntr;
    int rptfrm;
    int tff;
    int rndctrl;
    int pqindex;
    int halfqp;
    int pquantizer;
    int postproc;
    int condover;
    int transacfrm;
    int transacfrm2;
    int transdctab;
    int pqdiff;
    int abspq;
    int dquantfrm;
    vc1_dqprofile_e dqprofile;
    int dqsbedge;
    int dqdbedge;
    int dqbilevel;
    int mvrange;
    enum vc1_mvmode_e mvmode;
    enum vc1_mvmode_e mvmode2;
    int lumscale;
    int lumshift;
    int mvtab;
    int cbptab;
    int ttmbf;
    int ttfrm;
    int bfraction;
    vc1_fptype_e fptype;
    int numref;
    int reffield;
    int dmvrange;
    int intcompfield;
    int lumscale1;  //  type was char in ucode
    int lumshift1;  //  type was char in ucode
    int lumscale2;  //  type was char in ucode
    int lumshift2;  //  type was char in ucode
    int mbmodetab;
    int imvtab;
    int icbptab;
    int fourmvbptab;
    int fourmvswitch;
    int intcomp;
    int twomvbptab;
    // simple,main
    int rangeredfrm;

    // Fields from pdec_vc1_pic_s
    unsigned int   HistBufferSize;                  // in units of 256
    // frame buffers
    unsigned int   FrameStride[2];                  // [y_c]
    unsigned int   luma_top_offset;                 // offset of luma top field in units of 256
    unsigned int   luma_bot_offset;                 // offset of luma bottom field in units of 256
    unsigned int   luma_frame_offset;               // offset of luma frame in units of 256
    unsigned int   chroma_top_offset;               // offset of chroma top field in units of 256
    unsigned int   chroma_bot_offset;               // offset of chroma bottom field in units of 256
    unsigned int   chroma_frame_offset;             // offset of chroma frame in units of 256

    unsigned short CodedWidth;                      // entrypoint specific
    unsigned short CodedHeight;                     // entrypoint specific

    unsigned char  loopfilter;                      // entrypoint specific
    unsigned char  fastuvmc;                        // entrypoint specific
    unsigned char  output_memory_layout;            // picture specific
    unsigned char  ref_memory_layout[2];            // picture specific 0: fwd, 1: bwd
    unsigned char  reserved3[3];                    // for alignment

    nvdec_display_param_s displayPara;
    nvdec_pass2_otf_ext_s ssm;

} nvdec_vc1_pic_s;

// MPEG-2
typedef struct _nvdec_mpeg2_pic_s
{
    nvdec_pass2_otf_s encryption_params;
    unsigned char eos[16];
    unsigned char explicitEOSPresentFlag;
    unsigned char reserved0[3];
    unsigned int stream_len;
    unsigned int slice_count;

    // Driver may or may not use based upon need.
    // If 0 then default value of 1<<27 = 298ms @ 450MHz will be used in ucode.
    // Driver can send this value based upon resolution using the formula:
    // gptimer_timeout_value = 3 * (cycles required for one frame)
    unsigned int gptimer_timeout_value;

    // Fields from vld_mpeg2_seq_pic_info_s
    short FrameWidth;                   // actual frame width
    short FrameHeight;                  // actual frame height
    unsigned char picture_structure;    // 0 => Reserved, 1 => Top field, 2 => Bottom field, 3 => Frame picture. Table 6-14.
    unsigned char picture_coding_type;  // 0 => Forbidden, 1 => I, 2 => P, 3 => B, 4 => D (for MPEG-2). Table 6-12.
    unsigned char intra_dc_precision;   // 0 => 8 bits, 1=> 9 bits, 2 => 10 bits, 3 => 11 bits. Table 6-13.
    char frame_pred_frame_dct;          // as in section 6.3.10
    char concealment_motion_vectors;    // as in section 6.3.10
    char intra_vlc_format;              // as in section 6.3.10
    unsigned char tileFormat                 : 2 ;   // 0: TBL; 1: KBL; 2: Tile16x16
    unsigned char gob_height                 : 3 ;   // Set GOB height, 0: GOB_2, 1: GOB_4, 2: GOB_8, 3: GOB_16, 4: GOB_32 (NVDEC3 onwards)
    unsigned char reserverd_surface_format   : 3 ;

    char reserved1;                     // always 0
    char f_code[4];                  // as in section 6.3.10

    // Fields from pdec_mpeg2_picture_setup_s
    unsigned short PicWidthInMbs;
    unsigned short  FrameHeightInMbs;
    unsigned int pitch_luma;
    unsigned int pitch_chroma;
    unsigned int luma_top_offset;
    unsigned int luma_bot_offset;
    unsigned int luma_frame_offset;
    unsigned int chroma_top_offset;
    unsigned int chroma_bot_offset;
    unsigned int chroma_frame_offset;
    unsigned int HistBufferSize;
    unsigned short output_memory_layout;
    unsigned short alternate_scan;
    unsigned short secondfield;
    /******************************/
    // Got rid of the union kept for compatibility with NVDEC1.
    // Removed field mpeg2, and kept rounding type.
    // NVDEC1 ucode is not using the mpeg2 field, instead using codec type from the methods.
    // Rounding type should only be set for Divx3.11.
    unsigned short rounding_type;
    /******************************/
    unsigned int MbInfoSizeInBytes;
    unsigned int q_scale_type;
    unsigned int top_field_first;
    unsigned int full_pel_fwd_vector;
    unsigned int full_pel_bwd_vector;
    unsigned char quant_mat_8x8intra[64];
    unsigned char quant_mat_8x8nonintra[64];
    unsigned int ref_memory_layout[2]; //0:for fwd; 1:for bwd

    nvdec_display_param_s displayPara;
    nvdec_pass2_otf_ext_s ssm;

} nvdec_mpeg2_pic_s;

// MPEG-4
typedef struct _nvdec_mpeg4_pic_s
{
    nvdec_pass2_otf_s encryption_params;
    unsigned char eos[16];
    unsigned char explicitEOSPresentFlag;
    unsigned char reserved2[3];     // for alignment
    unsigned int stream_len;
    unsigned int slice_count;
    unsigned int scratch_pic_buffer_size;

    // Driver may or may not use based upon need.
    // If 0 then default value of 1<<27 = 298ms @ 450MHz will be used in ucode.
    // Driver can send this value based upon resolution using the formula:
    // gptimer_timeout_value = 3 * (cycles required for one frame)
    unsigned int gptimer_timeout_value;

    // Fields from vld_mpeg4_seq_s
    short FrameWidth;                     // :13 video_object_layer_width
    short FrameHeight;                    // :13 video_object_layer_height
    char  vop_time_increment_bitcount;    // : 5 1..16
    char  resync_marker_disable;          // : 1
    unsigned char tileFormat                 : 2 ;   // 0: TBL; 1: KBL; 2: Tile16x16
    unsigned char gob_height                 : 3 ;   // Set GOB height, 0: GOB_2, 1: GOB_4, 2: GOB_8, 3: GOB_16, 4: GOB_32 (NVDEC3 onwards)
    unsigned char reserverd_surface_format   : 3 ;
    char  reserved3;                      // for alignment

    // Fields from pdec_mpeg4_picture_setup_s
    int width;                              // : 13
    int height;                             // : 13

    unsigned int FrameStride[2];            // [y_c]
    unsigned int luma_top_offset;           // offset of luma top field in units of 256
    unsigned int luma_bot_offset;           // offset of luma bottom field in units of 256
    unsigned int luma_frame_offset;         // offset of luma frame in units of 256
    unsigned int chroma_top_offset;         // offset of chroma top field in units of 256
    unsigned int chroma_bot_offset;         // offset of chroma bottom field in units of 256
    unsigned int chroma_frame_offset;       // offset of chroma frame in units of 256

    unsigned int HistBufferSize;            // in units of 256, History buffer size

    int trd[2];                             // : 16, temporal reference frame distance (only needed for B-VOPs)
    int trb[2];                             // : 16, temporal reference B-VOP distance from fwd reference frame (only needed for B-VOPs)

    int divx_flags;                         // : 16 (bit 0: DivX interlaced chroma rounding, bit 1: Divx 4 boundary padding, bit 2: Divx IDCT)

    short vop_fcode_forward;                // : 1...7
    short vop_fcode_backward;               // : 1...7

    unsigned char interlaced;               // : 1
    unsigned char quant_type;               // : 1
    unsigned char quarter_sample;           // : 1
    unsigned char short_video_header;       // : 1

    unsigned char curr_output_memory_layout; // : 1 0:NV12; 1:NV24
    unsigned char ptype;                    // picture type: 0 for PTYPE_I, 1 for PTYPE_P, 2 for PTYPE_B, 3 for PTYPE_BI, 4 for PTYPE_SKIPPED
    unsigned char rnd;                      // : 1, rounding mode
    unsigned char alternate_vertical_scan_flag; // : 1

    unsigned char top_field_flag;           // : 1
    unsigned char reserved0[3];             // alignment purpose

    unsigned char intra_quant_mat[64];      // : 64*8
    unsigned char nonintra_quant_mat[64];   // : 64*8
    unsigned char ref_memory_layout[2];    //0:for fwd; 1:for bwd
    unsigned char reserved1[34];            // 256 byte alignemnt till now

    nvdec_display_param_s displayPara;

} nvdec_mpeg4_pic_s;

// VP8
enum VP8_FRAME_TYPE
{
    VP8_KEYFRAME = 0,
    VP8_INTERFRAME = 1
};

enum VP8_FRAME_SFC_ID
{
    VP8_GOLDEN_FRAME_SFC = 0,
    VP8_ALTREF_FRAME_SFC,
    VP8_LAST_FRAME_SFC,
    VP8_CURR_FRAME_SFC
};

typedef struct _nvdec_vp8_pic_s
{
    nvdec_pass2_otf_s encryption_params;

    // Driver may or may not use based upon need.
    // If 0 then default value of 1<<27 = 298ms @ 450MHz will be used in ucode.
    // Driver can send this value based upon resolution using the formula:
    // gptimer_timeout_value = 3 * (cycles required for one frame)
    unsigned int gptimer_timeout_value;

    unsigned short FrameWidth;     // actual frame width
    unsigned short FrameHeight;    // actual frame height

    unsigned char keyFrame;        // 1: key frame; 0: not
    unsigned char version;
    unsigned char tileFormat                 : 2 ;   // 0: TBL; 1: KBL; 2: Tile16x16
    unsigned char gob_height                 : 3 ;   // Set GOB height, 0: GOB_2, 1: GOB_4, 2: GOB_8, 3: GOB_16, 4: GOB_32 (NVDEC3 onwards)
    unsigned char reserverd_surface_format   : 3 ;
    unsigned char errorConcealOn;  // 1: error conceal on; 0: off

    unsigned int  firstPartSize;   // the size of first partition(frame header and mb header partition)

    // ctx
    unsigned int   HistBufferSize;                  // in units of 256
    unsigned int   VLDBufferSize;                   // in units of 1
    // current frame buffers
    unsigned int   FrameStride[2];                  // [y_c]
    unsigned int   luma_top_offset;                 // offset of luma top field in units of 256
    unsigned int   luma_bot_offset;                 // offset of luma bottom field in units of 256
    unsigned int   luma_frame_offset;               // offset of luma frame in units of 256
    unsigned int   chroma_top_offset;               // offset of chroma top field in units of 256
    unsigned int   chroma_bot_offset;               // offset of chroma bottom field in units of 256
    unsigned int   chroma_frame_offset;             // offset of chroma frame in units of 256

    nvdec_display_param_s displayPara;

    // decode picture buffere related
    char current_output_memory_layout;
    char output_memory_layout[3];  // output NV12/NV24 setting. item 0:golden; 1: altref; 2: last

    unsigned char segmentation_feature_data_update;
    unsigned char reserved1[3];

    // ucode return result
    unsigned int resultValue;      // ucode return the picture header info; includes copy_buffer_to_golden etc.
    unsigned int partition_offset[8];            // byte offset to each token partition (used for encrypted streams only)

    nvdec_pass2_otf_ext_s ssm;

} nvdec_vp8_pic_s; // size is 0xc0

// PASS1

//Sample means the entire frame is encrypted with a single IV, and subsample means a given frame may be encrypted in multiple chunks with different IVs.
#define NUM_SUBSAMPLES      32

typedef struct _bytes_of_data_s
{
    unsigned int    clear_bytes;                    // clear bytes per subsample
    unsigned int    encypted_bytes;                 // encrypted bytes per subsample

} bytes_of_data_s;

typedef struct _nvdec_pass1_input_data_s
{
    bytes_of_data_s sample_size[NUM_SUBSAMPLES];    // clear/encrypted bytes per subsample
    unsigned int    initialization_vector[NUM_SUBSAMPLES][4];   // Ctrl64 initial vector per subsample
    unsigned char   IvValid[NUM_SUBSAMPLES];        // each element will tell whether IV is valid for that subsample or not.
    unsigned int    stream_len;                     // encrypted bitstream size.
    unsigned int    clearBufferSize;                // allocated size of clear buffer size
    unsigned int    reencryptBufferSize;            // allocated size of reencrypted buffer size
    unsigned int    vp8coeffPartitonBufferSize;     // allocated buffer for vp8 coeff partition buffer
    unsigned int    PrevWidth;                        // required for VP9
    unsigned int    num_nals        :16;            // number of subsamples in a frame
    unsigned int    drm_mode        : 8;            // DRM mode
    unsigned int    key_sel         : 4;            // key select from keyslot
    unsigned int    codec           : 4;            // codecs selection
    unsigned int    TotalSizeOfClearData;           // Used with Pattern based encryption
    unsigned int    SliceHdrOffset;                 // This is used with pattern mode encryption where data before slice hdr comes in clear.
    unsigned int    EncryptBlkCnt   :16;
    unsigned int    SkipBlkCnt      :16;
} nvdec_pass1_input_data_s;

#define VP8_MAX_TOKEN_PARTITIONS     8
#define VP9_MAX_FRAMES_IN_SUPERFRAME 8

typedef struct _nvdec_pass1_output_data_s
{
    unsigned int    clear_header_size;              // h264/vc1/mpeg2/vp8, decrypted pps/sps/part of slice header info, 128 bits aligned
    unsigned int    reencrypt_data_size;            // h264/vc1/mpeg2, slice level data, vp8 mb header info, 128 bits aligned
    unsigned int    clear_token_data_size;          // vp8, clear token data saved in VPR, 128 bits aligned
    unsigned int    key_increment   : 6;            // added to content key after unwrapping
    unsigned int    encryption_mode : 4;            // encryption mode
    unsigned int    bReEncrypted    : 1;            // set to 0 if no re-encryption is done.
    unsigned int    bvp9SuperFrame  : 1;            // set to 1 for vp9 superframe
    unsigned int    vp9NumFramesMinus1    : 3;      // set equal to numFrames-1 for vp9superframe. Max 8 frames are possible in vp9 superframe.
    unsigned int    reserved1       :17;            // reserved, 32 bit alignment
    unsigned int    wrapped_session_key[4];         // session keys
    unsigned int    wrapped_content_key[4];         // content keys
    unsigned int    initialization_vector[4];       // Ctrl64 initial vector
    union {
        unsigned int    partition_size[VP8_MAX_TOKEN_PARTITIONS];            // size of each token partition (used for encrypted streams of VP8)
        unsigned int    vp9_frame_sizes[VP9_MAX_FRAMES_IN_SUPERFRAME];       // frame size information for all frames in vp9 superframe.
    };
    unsigned int    vp9_clear_hdr_size[VP9_MAX_FRAMES_IN_SUPERFRAME];          // clear header size for each frame in vp9 superframe.
} nvdec_pass1_output_data_s;


/*****************************************************
            AV1
*****************************************************/
typedef struct _scale_factors_reference_s{
  short             x_scale_fp;                                // horizontal fixed point scale factor
  short             y_scale_fp;                                // vertical fixed point scale factor
}scale_factors_reference_s;

typedef struct _frame_info_t{
    unsigned short  width;                                     // in pixel, av1 support arbitray resolution
    unsigned short  height;
    unsigned short  stride[2];                                 // luma and chroma stride in 16Bytes
    unsigned int    frame_buffer_idx;                          // TBD :clean associate the reference frame and frame buffer id to lookup base_addr
} frame_info_t;

typedef struct _ref_frame_struct_s{
    frame_info_t    info;
    scale_factors_reference_s sf;                              // scalefactor for reference frame and current frame size, driver can calculate it
    unsigned char   sign_bias                    : 1;          // calcuate based on frame_offset and current frame offset 
    unsigned char   wmtype                       : 2;          // global motion parameters : identity,translation,rotzoom,affine
    unsigned char   reserved_rf                  : 5;          
    short           frame_off;                                 // relative offset to current frame  
    short           roffset;                                   // relative offset from current frame  
} ref_frame_struct_s;

typedef struct _av1_fgs_cfg_t{
    //from AV1 spec 5.9.30 Film Grain Params syntax 
    unsigned short apply_grain                   : 1; 
    unsigned short overlap_flag                  : 1; 
    unsigned short clip_to_restricted_range      : 1; 
    unsigned short chroma_scaling_from_luma      : 1; 
    unsigned short num_y_points_b                : 1;          // flag indicates num_y_points>0 
    unsigned short num_cb_points_b               : 1;          // flag indicates num_cb_points>0 
    unsigned short num_cr_points_b               : 1;          // flag indicates num_cr_points>0
    unsigned short scaling_shift                 : 4;
    unsigned short reserved_fgs                  : 5;
	unsigned short sw_random_seed;
	short          cb_offset;
	short          cr_offset;
	char           cb_mult;
	char           cb_luma_mult;
	char           cr_mult;
	char           cr_luma_mult;
} av1_fgs_cfg_t;


typedef struct _nvdec_av1_pic_s
{
    nvdec_pass2_otf_s encryption_params;

    nvdec_pass2_otf_ext_s ssm;

    av1_fgs_cfg_t fgs_cfg;

    // Driver may or may not use based upon need.
    // If 0 then default value of 1<<27 = 298ms @ 450MHz will be used in ucode.
    // Driver can send this value based upon resolution using the formula:
    // gptimer_timeout_value = 3 * (cycles required for one frame)
    unsigned int    gptimer_timeout_value;

    unsigned int    stream_len;                                // stream length.
    unsigned int    reserved12;                                // skip bytes length to real frame data .

    //sequence header 
    unsigned int    use_128x128_superblock       : 1;          // superblock 128x128 or 64x64, 0:64x64, 1: 128x128
    unsigned int    chroma_format                : 2;          // 1:420, others:reserved for future
    unsigned int    bit_depth                    : 4;          // bitdepth
    unsigned int    enable_filter_intra          : 1;          // tool enable in seq level, 0 : disable 1: frame header control
    unsigned int    enable_intra_edge_filter     : 1;
    unsigned int    enable_interintra_compound   : 1;
    unsigned int    enable_masked_compound       : 1;
    unsigned int    enable_dual_filter           : 1;          // enable or disable vertical and horiz filter selection
    unsigned int    reserved10                   : 1;          // 0 - disable order hint, and related tools
    unsigned int    reserved0                    : 3;         
    unsigned int    enable_jnt_comp              : 1;          // 0 - disable joint compound modes
    unsigned int    reserved1                    : 1;
    unsigned int    enable_cdef                  : 1;
    unsigned int    reserved11                   : 1;
    unsigned int    enable_fgs                   : 1;
    unsigned int    enable_substream_decoding    : 1;          //enable frame substream kickoff mode without context switch 
    unsigned int    reserved2                    : 10;         // reserved bits

    //frame header     
    unsigned int    frame_type                   : 2;          // 0:Key frame, 1:Inter frame, 2:intra only, 3:s-frame
    unsigned int    show_frame                   : 1;          // show frame flag 
    unsigned int    reserved13                   : 1;  
    unsigned int    disable_cdf_update           : 1;          // disable CDF update during symbol decoding
    unsigned int    allow_screen_content_tools   : 1;          // screen content tool enable
    unsigned int    cur_frame_force_integer_mv   : 1;          // AMVR enable
    unsigned int    scale_denom_minus9           : 3;          // The denominator minus9  of the superres scale
    unsigned int    allow_intrabc                : 1;          // IBC enable
    unsigned int    allow_high_precision_mv      : 1;          // 1/8 precision mv enable 
    unsigned int    interp_filter                : 3;          // interpolation filter : EIGHTTAP_REGULAR,....
    unsigned int    switchable_motion_mode       : 1;          // 0: simple motion mode, 1: SIMPLE, OBMC, LOCAL  WARP
    unsigned int    use_ref_frame_mvs            : 1;          // 1: current frame can use the previous frame mv information, MFMV
    unsigned int    refresh_frame_context        : 1;          // backward update flag
    unsigned int    delta_q_present_flag         : 1;          // quantizer index delta values are present in the block level
    unsigned int    delta_q_res                  : 2;          // left shift will apply to decoded quantizer index delta values
    unsigned int    delta_lf_present_flag        : 1;          // specified whether loop filter delta values are present in the block level
    unsigned int    delta_lf_res                 : 2;          // specifies the left shift will apply to decoded loop filter values
    unsigned int    delta_lf_multi               : 1;          // seperate loop filter deltas for Hy,Vy,U,V edges
    unsigned int    reserved3                    : 1;         
    unsigned int    coded_lossless               : 1;          // 1 means all segments use lossless coding. Frame is fully lossless, CDEF/DBF will disable 
    unsigned int    tile_enabled                 : 1;          // tile enable  
    unsigned int    reserved4                    : 2;        
    unsigned int    superres_is_scaled           : 1;          // frame level frame for using_superres                
    unsigned int    reserved_fh                  : 1;
    
    unsigned int    tile_cols                    : 8;          // horizontal tile numbers in frame, max is 64
    unsigned int    tile_rows                    : 8;          // vertical tile numbers in frame, max is 64
    unsigned int    context_update_tile_id       : 16;         // which tile cdf will be seleted as the backward update CDF, MAXTILEROW=64, MAXTILECOL=64, 12bits
    
    unsigned int    cdef_damping_minus_3         : 2;          // controls the amount of damping in the deringing filter 
    unsigned int    cdef_bits                    : 2;          // the number of bits needed to specify which CDEF filter to apply    
    unsigned int    frame_tx_mode                : 3;          // 0:ONLY4x4,3:LARGEST,4:SELECT
    unsigned int    frame_reference_mode         : 2;          // single,compound,select
    unsigned int    skip_mode_flag               : 1;          // skip mode
    unsigned int    skip_ref0                    : 4;  
    unsigned int    skip_ref1                    : 4;  
    unsigned int    allow_warp                   : 1;          // sequence level & frame level warp enable
    unsigned int    reduced_tx_set_used          : 1;          // whether the frame is  restricted to oa reduced subset of the full set of transform types
    unsigned int    ref_scaling_enable           : 1;
    unsigned int    reserved5                    : 1;            
    unsigned int    reserved6                    : 10;         // reserved bits                 
    unsigned short  superres_upscaled_width;                   // upscale width, frame_size_with_refs() syntax,restoration will use it
    unsigned short  superres_luma_step;
    unsigned short  superres_chroma_step;
    unsigned short  superres_init_luma_subpel_x;
    unsigned short  superres_init_chroma_subpel_x;

    /*frame header qp information*/
    unsigned char   base_qindex;                               // the maximum qp is 255
    char            y_dc_delta_q;
    char            u_dc_delta_q;
    char            v_dc_delta_q;
    char            u_ac_delta_q;
    char            v_ac_delta_q;
    unsigned char   qm_y;                                      // 4bit: 0-15
    unsigned char   qm_u;
    unsigned char   qm_v;
 
    /*cdef, need to update in the new spec*/
    unsigned int    cdef_y_pri_strength;                       // 4bit for one, max is 8
    unsigned int    cdef_uv_pri_strength;                      // 4bit for one, max is 8
    unsigned int    cdef_y_sec_strength          : 16;         // 2bit for one, max is 8
    unsigned int    cdef_uv_sec_strength         : 16;         // 2bit for one, max is 8

    /*segmentation*/
    unsigned char   segment_enabled;
    unsigned char   segment_update_map;
    unsigned char   reserved7;
    unsigned char   segment_temporal_update;
    short           segment_feature_data[8][8];
    unsigned char   last_active_segid;                         // The highest numbered segment id that has some enabled feature.
    unsigned char   segid_preskip;                             // Whether the segment id will be read before the skip syntax element.
                                                               // 1: the segment id will be read first.
                                                               // 0: the skip syntax element will be read first.
    unsigned char   prevsegid_flag;                            // 1 : previous segment id is  available 
    unsigned char   segment_quant_sign           : 8;          // sign bit for segment alternative QP  

    /*loopfilter*/
    unsigned char   filter_level[2];
    unsigned char   filter_level_u;
    unsigned char   filter_level_v;
    unsigned char   lf_sharpness_level;
    char            lf_ref_deltas[8];                          // 0 = Intra, Last, Last2+Last3, GF, BRF, ARF2, ARF
    char            lf_mode_deltas[2];                         // 0 = ZERO_MV, MV

    /*restoration*/
    unsigned char   lr_type ;                                  // restoration type.  Y:bit[1:0];U:bit[3:2],V:bit[5:4]  
    unsigned char   lr_unit_size;                              // restoration unit size 0:32x32, 1:64x64, 2:128x128,3:256x256;  Y:bit[1:0];U:bit[3:2],V:bit[5:4]

    //general
    frame_info_t    current_frame;
    ref_frame_struct_s ref_frame[7];                           // Last, Last2, Last3, Golden, BWDREF, ALTREF2, ALTREF 
    
    unsigned int    use_temporal0_mvs            : 1;
    unsigned int    use_temporal1_mvs            : 1;
    unsigned int    use_temporal2_mvs            : 1;
    unsigned int    mf1_type                     : 3; 
    unsigned int    mf2_type                     : 3; 
    unsigned int    mf3_type                     : 3; 
    unsigned int    reserved_mfmv                : 20;

    short           mfmv_offset[3][7];                         // 3: mf0~2, 7: Last, Last2, Last3, Golden, BWDREF, ALTREF2, ALTREF
    char            mfmv_side[3][7];                           // flag for reverse offset great than 0
                                                               // MFMV relative offset from the ref frame(reference to reference relative offset)

    unsigned char   tileformat                   : 2;          // 0: TBL; 1: KBL;
    unsigned char   gob_height                   : 3;          // Set GOB height, 0: GOB_2, 1: GOB_4, 2: GOB_8, 3: GOB_16, 4: GOB_32 (NVDEC3 onwards)
    unsigned char   errorConcealOn               : 1;          // this field is not used, use ctrl_param.error_conceal_on to enable error concealment in ucode, 
                                                               // always set NV_CNVDEC_GIP_ERR_CONCEAL_CTRL_ON = 1 to enable error detect in hw
    unsigned char   reserver8                    : 2;          // reserve 
    
    unsigned char   stream_error_detection       : 1;
    unsigned char   mv_error_detection           : 1;
    unsigned char   coeff_error_detection        : 1;
    unsigned char   reserved_eh                  : 5;

    // Filt neighbor buffer offset
    unsigned int    Av1FltTopOffset;                           // filter top buffer offset respect to filter buffer, 256 bytes unit
    unsigned int    Av1FltVertOffset;                          // filter vertical buffer offset respect to filter buffer, 256 bytes unit
    unsigned int    Av1CdefVertOffset;                         // cdef vertical buffer offset respect to filter buffer, 256 bytes unit
    unsigned int    Av1LrVertOffset;                           // lr vertical buffer offset respect to filter buffer, 256 bytes unit
    unsigned int    Av1HusVertOffset;                          // hus vertical buffer offset respect to filter buffer, 256 bytes unit
    unsigned int    Av1FgsVertOffset;                          // fgs vertical buffer offset respect to filter buffer, 256 bytes unit
    
    unsigned int    enable_histogram             : 1;
    unsigned int    sw_skip_start_length         : 14;         //skip start length 
    unsigned int    reserved_stat                : 17;

} nvdec_av1_pic_s;

//////////////////////////////////////////////////////////////////////
// AV1 Buffer structure
//////////////////////////////////////////////////////////////////////
typedef struct _AV1FilmGrainMemory 
 {
    unsigned char   scaling_lut_y[256];
    unsigned char   scaling_lut_cb[256];
    unsigned char   scaling_lut_cr[256];
    short           cropped_luma_grain_block[4096];
    short           cropped_cb_grain_block[1024];
    short           cropped_cr_grain_block[1024];
} AV1FilmGrainMemory;

typedef struct _AV1TileInfo_OLD
{
    unsigned char   width_in_sb;
    unsigned char   height_in_sb;
    unsigned char   tile_start_b0;
    unsigned char   tile_start_b1;
    unsigned char   tile_start_b2;
    unsigned char   tile_start_b3;
    unsigned char   tile_end_b0;
    unsigned char   tile_end_b1;
    unsigned char   tile_end_b2;
    unsigned char   tile_end_b3;
    unsigned char   padding[6];
} AV1TileInfo_OLD; 

typedef struct _AV1TileInfo
{
    unsigned char   width_in_sb;
    unsigned char   padding_w;
    unsigned char   height_in_sb;
    unsigned char   padding_h;
} AV1TileInfo; 

typedef struct _AV1TileStreamInfo
{
    unsigned int    tile_start;
    unsigned int    tile_end;
    unsigned char   padding[8];
} AV1TileStreamInfo; 


// AV1 TileSize buffer
#define AV1_MAX_TILES                       256
#define AV1_TILEINFO_BUF_SIZE_OLD           NVDEC_ALIGN(AV1_MAX_TILES * sizeof(AV1TileInfo_OLD))
#define AV1_TILEINFO_BUF_SIZE               NVDEC_ALIGN(AV1_MAX_TILES * sizeof(AV1TileInfo))

// AV1 TileStreamInfo buffer
#define AV1_TILESTREAMINFO_BUF_SIZE         NVDEC_ALIGN(AV1_MAX_TILES * sizeof(AV1TileStreamInfo))

// AV1 SubStreamEntry buffer
#define MAX_SUBSTREAM_ENTRY_SIZE            32
#define AV1_SUBSTREAM_ENTRY_BUF_SIZE        NVDEC_ALIGN(MAX_SUBSTREAM_ENTRY_SIZE * sizeof(nvdec_substream_entry_s))

// AV1 FilmGrain Parameter buffer
#define AV1_FGS_BUF_SIZE                    NVDEC_ALIGN(sizeof(AV1FilmGrainMemory))

// AV1 Temporal MV buffer
#define AV1_TEMPORAL_MV_SIZE_IN_64x64       256            // 4Bytes for 8x8
#define AV1_TEMPORAL_MV_BUF_SIZE(w, h)      ALIGN_UP( ALIGN_UP(w,128) * ALIGN_UP(h,128) / (64*64) * AV1_TEMPORAL_MV_SIZE_IN_64x64, 4096)            

// AV1 SegmentID buffer
#define AV1_SEGMENT_ID_SIZE_IN_64x64        128            // (3bits + 1 pad_bits) for 4x4
#define AV1_SEGMENT_ID_BUF_SIZE(w, h)       ALIGN_UP( ALIGN_UP(w,128) * ALIGN_UP(h,128) / (64*64) * AV1_SEGMENT_ID_SIZE_IN_64x64, 4096)            

// AV1 Global Motion buffer 
#define AV1_GLOBAL_MOTION_BUF_SIZE          NVDEC_ALIGN(7*32)

// AV1 Intra Top buffer
#define AV1_INTRA_TOP_BUF_SIZE              NVDEC_ALIGN(8*8192)

// AV1 Histogram buffer
#define AV1_HISTOGRAM_BUF_SIZE              NVDEC_ALIGN(1024)

// AV1 Filter FG buffer
#define AV1_DBLK_TOP_SIZE_IN_SB64           ALIGN_UP(1920, 128)                
#define AV1_DBLK_TOP_BUF_SIZE(w)            NVDEC_ALIGN( (ALIGN_UP(w,64)/64 + 2) * AV1_DBLK_TOP_SIZE_IN_SB64)

#define AV1_DBLK_LEFT_SIZE_IN_SB64          ALIGN_UP(1536, 128)                
#define AV1_DBLK_LEFT_BUF_SIZE(h)           NVDEC_ALIGN( (ALIGN_UP(h,64)/64 + 2) * AV1_DBLK_LEFT_SIZE_IN_SB64)

#define AV1_CDEF_LEFT_SIZE_IN_SB64          ALIGN_UP(1792, 128)                
#define AV1_CDEF_LEFT_BUF_SIZE(h)           NVDEC_ALIGN( (ALIGN_UP(h,64)/64 + 2) * AV1_CDEF_LEFT_SIZE_IN_SB64)

#define AV1_HUS_LEFT_SIZE_IN_SB64           ALIGN_UP(12544, 128) 
#define AV1_ASIC_HUS_LEFT_BUFFER_SIZE(h)    NVDEC_ALIGN( (ALIGN_UP(h,64)/64 + 2) * AV1_HUS_LEFT_SIZE_IN_SB64)
#define AV1_HUS_LEFT_BUF_SIZE(h)            2*AV1_ASIC_HUS_LEFT_BUFFER_SIZE(h)     // Ping-Pong buffers

#define AV1_LR_LEFT_SIZE_IN_SB64            ALIGN_UP(1920, 128)                
#define AV1_LR_LEFT_BUF_SIZE(h)             NVDEC_ALIGN( (ALIGN_UP(h,64)/64 + 2) * AV1_LR_LEFT_SIZE_IN_SB64)

#define AV1_FGS_LEFT_SIZE_IN_SB64           ALIGN_UP(320, 128)                
#define AV1_FGS_LEFT_BUF_SIZE(h)            NVDEC_ALIGN( (ALIGN_UP(h,64)/64 + 2) * AV1_FGS_LEFT_SIZE_IN_SB64)

// AV1 Hint Dump Buffer
#define AV1_HINT_DUMP_SIZE_IN_SB64          ((64*64)/(4*4)*8)           // 8 bytes per CU, 256 CUs(2048 bytes) per SB64
#define AV1_HINT_DUMP_SIZE_IN_SB128         ((128*128)/(4*4)*8)         // 8 bytes per CU,1024 CUs(8192 bytes) per SB128
#define AV1_HINT_DUMP_SIZE(w, h)            NVDEC_ALIGN(AV1_HINT_DUMP_SIZE_IN_SB128*((w+127)/128)*((h+127)/128))  // always use SB128 for allocation


/*******************************************************************
                New  H264 
********************************************************************/
typedef struct _nvdec_new_h264_pic_s
{
    nvdec_pass2_otf_s encryption_params;
    unsigned char eos[16];
    unsigned char explicitEOSPresentFlag;
    unsigned char hint_dump_en; //enable COLOMV surface dump for all frames, which includes hints of "MV/REFIDX/QP/CBP/MBPART/MBTYPE", nvbug: 200212874
    unsigned char reserved0[2];
    unsigned int stream_len;
    unsigned int slice_count;
    unsigned int mbhist_buffer_size;     // to pass buffer size of MBHIST_BUFFER

    // Driver may or may not use based upon need.
    // If 0 then default value of 1<<27 = 298ms @ 450MHz will be used in ucode.
    // Driver can send this value based upon resolution using the formula:
    // gptimer_timeout_value = 3 * (cycles required for one frame)
    unsigned int gptimer_timeout_value;

    // Fields from msvld_h264_seq_s
    int log2_max_pic_order_cnt_lsb_minus4;
    int delta_pic_order_always_zero_flag;
    int frame_mbs_only_flag;
    int PicWidthInMbs;
    int FrameHeightInMbs;

    unsigned int tileFormat                 : 2 ;   // 0: TBL; 1: KBL; 2: Tile16x16
    unsigned int gob_height                 : 3 ;   // Set GOB height, 0: GOB_2, 1: GOB_4, 2: GOB_8, 3: GOB_16, 4: GOB_32 (NVDEC3 onwards)
    unsigned int reserverd_surface_format   : 27;

    // Fields from msvld_h264_pic_s
    int entropy_coding_mode_flag;
    int pic_order_present_flag;
    int num_ref_idx_l0_active_minus1;
    int num_ref_idx_l1_active_minus1;
    int deblocking_filter_control_present_flag;
    int redundant_pic_cnt_present_flag;
    int transform_8x8_mode_flag;

    // Fields from mspdec_h264_picture_setup_s
    unsigned int pitch_luma;                    // Luma pitch
    unsigned int pitch_chroma;                  // chroma pitch

    unsigned int luma_top_offset;               // offset of luma top field in units of 256
    unsigned int luma_bot_offset;               // offset of luma bottom field in units of 256
    unsigned int luma_frame_offset;             // offset of luma frame in units of 256
    unsigned int chroma_top_offset;             // offset of chroma top field in units of 256
    unsigned int chroma_bot_offset;             // offset of chroma bottom field in units of 256
    unsigned int chroma_frame_offset;           // offset of chroma frame in units of 256
    unsigned int HistBufferSize;                // in units of 256

    unsigned int MbaffFrameFlag           : 1;  //
    unsigned int direct_8x8_inference_flag: 1;  //
    unsigned int weighted_pred_flag       : 1;  //
    unsigned int constrained_intra_pred_flag:1; //
    unsigned int ref_pic_flag             : 1;  // reference picture (nal_ref_idc != 0)
    unsigned int field_pic_flag           : 1;  //
    unsigned int bottom_field_flag        : 1;  //
    unsigned int second_field             : 1;  // second field of complementary reference field
    unsigned int log2_max_frame_num_minus4: 4;  //  (0..12)
    unsigned int chroma_format_idc        : 2;  //
    unsigned int pic_order_cnt_type       : 2;  //  (0..2)
    int pic_init_qp_minus26               : 6;  // : 6 (-26..+25)
    int chroma_qp_index_offset            : 5;  // : 5 (-12..+12)
    int second_chroma_qp_index_offset     : 5;  // : 5 (-12..+12)

    unsigned int weighted_bipred_idc      : 2;  // : 2 (0..2)
    unsigned int CurrPicIdx               : 7;  // : 7  uncompressed frame buffer index
    unsigned int CurrColIdx               : 5;  // : 5  index of associated co-located motion data buffer
    unsigned int frame_num                : 16; //
    unsigned int frame_surfaces           : 1;  // frame surfaces flag
    unsigned int output_memory_layout     : 1;  // 0: NV12; 1:NV24. Field pair must use the same setting.

    int CurrFieldOrderCnt[2];                   // : 32 [Top_Bottom], [0]=TopFieldOrderCnt, [1]=BottomFieldOrderCnt
    nvdec_dpb_entry_s dpb[16];
    unsigned char WeightScale[6][4][4];         // : 6*4*4*8 in raster scan order (not zig-zag order)
    unsigned char WeightScale8x8[2][8][8];      // : 2*8*8*8 in raster scan order (not zig-zag order)

    // mvc setup info, must be zero if not mvc
    unsigned char num_inter_view_refs_lX[2];         // number of inter-view references
    char reserved1[14];                               // reserved for alignment
    signed char inter_view_refidx_lX[2][16];         // DPB indices (must also be marked as long-term)

    // lossless decode (At the time of writing this manual, x264 and JM encoders, differ in Intra_8x8 reference sample filtering)
    unsigned int lossless_ipred8x8_filter_enable        : 1;       // = 0, skips Intra_8x8 reference sample filtering, for vertical and horizontal predictions (x264 encoded streams); = 1, filter Intra_8x8 reference samples (JM encoded streams)
    unsigned int qpprime_y_zero_transform_bypass_flag   : 1;       // determines the transform bypass mode
    unsigned int reserved2                              : 30;      // kept for alignment; may be used for other parameters

    nvdec_display_param_s displayPara;
    nvdec_pass2_otf_ext_s ssm;

} nvdec_new_h264_pic_s;

// golden crc struct dumped into surface
// for each part, if golden crc compare is enabled, one interface is selected to do crc calculation in vmod.
// vmod's crc is compared with cmod's golden crc (4*32 bits), and compare reuslt is written into surface.
typedef struct
{
    // input
    unsigned int    dbg_crc_enable_partb    : 1;    // Eable flag for enable/disable interface crc calculation in NVDEC HW's part b
    unsigned int    dbg_crc_enable_partc    : 1;    // Eable flag for enable/disable interface crc calculation in NVDEC HW's part c
    unsigned int    dbg_crc_enable_partd    : 1;    // Eable flag for enable/disable interface crc calculation in NVDEC HW's part d
    unsigned int    dbg_crc_enable_parte    : 1;    // Eable flag for enable/disable interface crc calculation in NVDEC HW's part e
    unsigned int    dbg_crc_intf_partb      : 6;    // For partb to select which interface to compare crc. see DBG_CRC_PARTE_INTF_SEL for detailed control value for each interface
    unsigned int    dbg_crc_intf_partc      : 6;    // For partc to select which interface to compare crc. see DBG_CRC_PARTE_INTF_SEL for detailed control value for each interface
    unsigned int    dbg_crc_intf_partd      : 6;    // For partd to select which interface to compare crc. see DBG_CRC_PARTE_INTF_SEL for detailed control value for each interface
    unsigned int    dbg_crc_intf_parte      : 6;    // For parte to select which interface to compare crc. see DBG_CRC_PARTE_INTF_SEL for detailed control value for each interface
    unsigned int    reserved0               : 4;

    unsigned int    dbg_crc_partb_golden[4];        // Golden crc values for part b
    unsigned int    dbg_crc_partc_golden[4];        // Golden crc values for part c
    unsigned int    dbg_crc_partd_golden[4];        // Golden crc values for part d
    unsigned int    dbg_crc_parte_golden[4];        // Golden crc values for part e

    // output
    unsigned int    dbg_crc_comp_partb      : 4;    // Compare result for part b
    unsigned int    dbg_crc_comp_partc      : 4;    // Compare result for part c
    unsigned int    dbg_crc_comp_partd      : 4;    // Compare result for part d
    unsigned int    dbg_crc_comp_parte      : 4;    // Compare result for part e
    unsigned int    reserved1               : 16;

    unsigned char   reserved2[56];
}nvdec_crc_s;                                       // 128 Bytes

#endif // __DRV_NVDEC_H_