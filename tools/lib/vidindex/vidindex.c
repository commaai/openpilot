#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <assert.h>

#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>

#include "./bitstream.h"

#define START_CODE 0x000001

static uint32_t read24be(const uint8_t* ptr) {
    return (ptr[0] << 16) | (ptr[1] << 8) | ptr[2];
}
static void write32le(FILE *of, uint32_t v) {
  uint8_t va[4] = {
    v & 0xff, (v >> 8) & 0xff, (v >> 16) & 0xff, (v >> 24) & 0xff
  };
  fwrite(va, 1, sizeof(va), of);
}

// Table 7-1
enum hevc_nal_type {
  HEVC_NAL_TYPE_TRAIL_N = 0,
  HEVC_NAL_TYPE_TRAIL_R = 1,
  HEVC_NAL_TYPE_TSA_N = 2,
  HEVC_NAL_TYPE_TSA_R = 3,
  HEVC_NAL_TYPE_STSA_N = 4,
  HEVC_NAL_TYPE_STSA_R = 5,
  HEVC_NAL_TYPE_RADL_N = 6,
  HEVC_NAL_TYPE_RADL_R = 7,
  HEVC_NAL_TYPE_RASL_N = 8,
  HEVC_NAL_TYPE_RASL_R = 9,
  HEVC_NAL_TYPE_BLA_W_LP = 16,
  HEVC_NAL_TYPE_BLA_W_RADL = 17,
  HEVC_NAL_TYPE_BLA_N_LP = 18,
  HEVC_NAL_TYPE_IDR_W_RADL = 19,
  HEVC_NAL_TYPE_IDR_N_LP = 20,
  HEVC_NAL_TYPE_CRA_NUT = 21,
  HEVC_NAL_TYPE_RSV_IRAP_VCL23 = 23,
  HEVC_NAL_TYPE_VPS_NUT = 32,
  HEVC_NAL_TYPE_SPS_NUT = 33,
  HEVC_NAL_TYPE_PPS_NUT = 34,
  HEVC_NAL_TYPE_AUD_NUT = 35,
  HEVC_NAL_TYPE_EOS_NUT = 36,
  HEVC_NAL_TYPE_EOB_NUT = 37,
  HEVC_NAL_TYPE_FD_NUT = 38,
  HEVC_NAL_TYPE_PREFIX_SEI_NUT = 39,
  HEVC_NAL_TYPE_SUFFIX_SEI_NUT = 40,
};

// Table 7-7
enum hevc_slice_type {
  HEVC_SLICE_B = 0,
  HEVC_SLICE_P = 1,
  HEVC_SLICE_I = 2,
};

static void hevc_index(const uint8_t *data, size_t file_size, FILE *of_prefix, FILE *of_index) {
  const uint8_t* ptr = data;
  const uint8_t* ptr_end = data + file_size;

  assert(ptr[0] == 0);
  ptr++;
  assert(read24be(ptr) == START_CODE);

  // pps. ignore for now
  uint32_t num_extra_slice_header_bits = 0;
  uint32_t dependent_slice_segments_enabled_flag = 0;

  while (ptr < ptr_end) {
    const uint8_t* next = ptr+1;
    for (; next < ptr_end-4; next++) {
      if (read24be(next) == START_CODE) break;
    }
    size_t nal_size = next - ptr;
    if (nal_size < 6) {
      break;
    }

    {
      struct bitstream bs = {0};
      bs_init(&bs, ptr, nal_size);

      uint32_t start_code = bs_get(&bs, 24);
      assert(start_code == 0x000001);

      // nal_unit_header
      uint32_t forbidden_zero_bit = bs_get(&bs, 1);
      uint32_t nal_unit_type = bs_get(&bs, 6);
      uint32_t nuh_layer_id = bs_get(&bs, 6);
      uint32_t nuh_temporal_id_plus1 = bs_get(&bs, 3);

      // if (nal_unit_type != 1) printf("%3d -- %3d %10d %lu\n", nal_unit_type, frame_num, (uint32_t)(ptr-data), nal_size);

      switch (nal_unit_type) {
      case HEVC_NAL_TYPE_VPS_NUT:
      case HEVC_NAL_TYPE_SPS_NUT:
      case HEVC_NAL_TYPE_PPS_NUT:
        fwrite(ptr, 1, nal_size, of_prefix);
        break;
      case HEVC_NAL_TYPE_TRAIL_N:
      case HEVC_NAL_TYPE_TRAIL_R:
      case HEVC_NAL_TYPE_TSA_N:
      case HEVC_NAL_TYPE_TSA_R:
      case HEVC_NAL_TYPE_STSA_N:
      case HEVC_NAL_TYPE_STSA_R:
      case HEVC_NAL_TYPE_RADL_N:
      case HEVC_NAL_TYPE_RADL_R:
      case HEVC_NAL_TYPE_RASL_N:
      case HEVC_NAL_TYPE_RASL_R:
      case HEVC_NAL_TYPE_BLA_W_LP:
      case HEVC_NAL_TYPE_BLA_W_RADL:
      case HEVC_NAL_TYPE_BLA_N_LP:
      case HEVC_NAL_TYPE_IDR_W_RADL:
      case HEVC_NAL_TYPE_IDR_N_LP:
      case HEVC_NAL_TYPE_CRA_NUT: {
        // slice_segment_header
        uint32_t first_slice_segment_in_pic_flag = bs_get(&bs, 1);
        if (nal_unit_type >= HEVC_NAL_TYPE_BLA_W_LP && nal_unit_type <= HEVC_NAL_TYPE_RSV_IRAP_VCL23) {
          uint32_t no_output_of_prior_pics_flag = bs_get(&bs, 1);
        }
        uint32_t slice_pic_parameter_set_id = bs_get(&bs, 1);
        if (!first_slice_segment_in_pic_flag) {
          // ...
          break;
        }

        if (!dependent_slice_segments_enabled_flag) {
          for (int i=0; i<num_extra_slice_header_bits; i++) {
            bs_get(&bs, 1);
          }
          uint32_t slice_type = bs_ue(&bs);

          // write the index
          write32le(of_index, slice_type);
          write32le(of_index, ptr - data);

          // ...
        }

        break;
      }
      }

      //...
      // emulation_prevention_three_byte
    }

    ptr = next;
  }

  write32le(of_index, -1);
  write32le(of_index, file_size);
}

// Table 7-1
enum h264_nal_type {
  H264_NAL_SLICE           = 1,
  H264_NAL_DPA             = 2,
  H264_NAL_DPB             = 3,
  H264_NAL_DPC             = 4,
  H264_NAL_IDR_SLICE       = 5,
  H264_NAL_SEI             = 6,
  H264_NAL_SPS             = 7,
  H264_NAL_PPS             = 8,
  H264_NAL_AUD             = 9,
  H264_NAL_END_SEQUENCE    = 10,
  H264_NAL_END_STREAM      = 11,
  H264_NAL_FILLER_DATA     = 12,
  H264_NAL_SPS_EXT         = 13,
  H264_NAL_AUXILIARY_SLICE = 19,
};

enum h264_slice_type {
  H264_SLICE_P = 0,
  H264_SLICE_B = 1,
  H264_SLICE_I = 2,
  // ...
};

static void h264_index(const uint8_t *data, size_t file_size, FILE *of_prefix, FILE *of_index) {
  const uint8_t* ptr = data;
  const uint8_t* ptr_end = data + file_size;

  assert(ptr[0] == 0);
  ptr++;
  assert(read24be(ptr) == START_CODE);


  uint32_t sps_log2_max_frame_num_minus4;


  int last_frame_num = -1;

  while (ptr < ptr_end) {
    const uint8_t* next = ptr+1;
    for (; next < ptr_end-4; next++) {
      if (read24be(next) == START_CODE) break;
    }
    size_t nal_size = next - ptr;
    if (nal_size < 5) {
      break;
    }

    {
      struct bitstream bs = {0};
      bs_init(&bs, ptr, nal_size);

      uint32_t start_code = bs_get(&bs, 24);
      assert(start_code == 0x000001);

      // nal_unit_header
      uint32_t forbidden_zero_bit = bs_get(&bs, 1);
      uint32_t nal_ref_idx = bs_get(&bs, 2);
      uint32_t nal_unit_type = bs_get(&bs, 5);

      switch (nal_unit_type) {
      case H264_NAL_SPS:

        {
          uint32_t profile_idx = bs_get(&bs, 8);
          uint32_t constraint_sets = bs_get(&bs, 4);
          uint32_t reserved = bs_get(&bs, 5);
          uint32_t level_idc = bs_get(&bs, 5);
          uint32_t seq_parameter_set_id = bs_ue(&bs);
          sps_log2_max_frame_num_minus4 = bs_ue(&bs);
        }

        // fallthrough
      case H264_NAL_PPS:
        fwrite(ptr, 1, nal_size, of_prefix);
        break;

      case H264_NAL_SLICE:
      case H264_NAL_IDR_SLICE: {
        // slice header
        uint32_t first_mb_in_slice = bs_ue(&bs);
        uint32_t slice_type = bs_ue(&bs);
        uint32_t pic_parameter_set_id = bs_ue(&bs);

        uint32_t frame_num = bs_get(&bs, sps_log2_max_frame_num_minus4+4);

        if (first_mb_in_slice == 0) {
          write32le(of_index, slice_type);
          write32le(of_index, ptr - data);
        }

        break;
      }

      }

    }

    ptr = next;
  }

  write32le(of_index, -1);
  write32le(of_index, file_size);
}

int main(int argc, char** argv) {
  if (argc != 5) {
    fprintf(stderr, "usage: %s h264|hevc file_path out_prefix out_index\n", argv[0]);
    exit(1);
  }

  const char* file_type = argv[1];
  const char* file_path = argv[2];

  int fd = open(file_path, O_RDONLY, 0);
  if (fd < 0) {
    fprintf(stderr, "error: couldn't open %s\n", file_path);
    exit(1);
  }

  FILE *of_prefix = fopen(argv[3], "wb");
  assert(of_prefix);
  FILE *of_index = fopen(argv[4], "wb");
  assert(of_index);

  off_t file_size = lseek(fd, 0, SEEK_END);
  lseek(fd, 0, SEEK_SET);

  assert(file_size > 4);

  const uint8_t* data = (const uint8_t*)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  assert(data != MAP_FAILED);

  if (strcmp(file_type, "hevc") == 0) {
    hevc_index(data, file_size, of_prefix, of_index);
  } else if (strcmp(file_type, "h264") == 0) {
    h264_index(data, file_size, of_prefix, of_index);
  } else {
    assert(false);
  }

  munmap((void*)data, file_size);
  close(fd);

  return 0;
}
