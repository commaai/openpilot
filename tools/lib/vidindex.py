#!/usr/bin/env python3
import argparse
import os
import struct
from enum import IntEnum

from openpilot.tools.lib.filereader import FileReader

DEBUG = int(os.getenv("DEBUG", "0"))

# compare to ffmpeg parsing
# ffmpeg -i <input.hevc> -c copy -bsf:v trace_headers -f null - 2>&1 | grep -B4 -A32 '] 0 '

# H.265 specification
# https://www.itu.int/rec/dologin_pub.asp?lang=e&id=T-REC-H.265-201802-S!!PDF-E&type=items

NAL_UNIT_START_CODE = b"\x00\x00\x01"
NAL_UNIT_START_CODE_SIZE = len(NAL_UNIT_START_CODE)
NAL_UNIT_HEADER_SIZE = 2

class HevcNalUnitType(IntEnum):
  TRAIL_N = 0         # RBSP structure: slice_segment_layer_rbsp( )
  TRAIL_R = 1         # RBSP structure: slice_segment_layer_rbsp( )
  TSA_N = 2           # RBSP structure: slice_segment_layer_rbsp( )
  TSA_R = 3           # RBSP structure: slice_segment_layer_rbsp( )
  STSA_N = 4          # RBSP structure: slice_segment_layer_rbsp( )
  STSA_R = 5          # RBSP structure: slice_segment_layer_rbsp( )
  RADL_N = 6          # RBSP structure: slice_segment_layer_rbsp( )
  RADL_R = 7          # RBSP structure: slice_segment_layer_rbsp( )
  RASL_N = 8          # RBSP structure: slice_segment_layer_rbsp( )
  RASL_R = 9          # RBSP structure: slice_segment_layer_rbsp( )
  RSV_VCL_N10 = 10
  RSV_VCL_R11 = 11
  RSV_VCL_N12 = 12
  RSV_VCL_R13 = 13
  RSV_VCL_N14 = 14
  RSV_VCL_R15 = 15
  BLA_W_LP = 16       # RBSP structure: slice_segment_layer_rbsp( )
  BLA_W_RADL = 17     # RBSP structure: slice_segment_layer_rbsp( )
  BLA_N_LP = 18       # RBSP structure: slice_segment_layer_rbsp( )
  IDR_W_RADL = 19     # RBSP structure: slice_segment_layer_rbsp( )
  IDR_N_LP = 20       # RBSP structure: slice_segment_layer_rbsp( )
  CRA_NUT = 21        # RBSP structure: slice_segment_layer_rbsp( )
  RSV_IRAP_VCL22 = 22
  RSV_IRAP_VCL23 = 23
  RSV_VCL24 = 24
  RSV_VCL25 = 25
  RSV_VCL26 = 26
  RSV_VCL27 = 27
  RSV_VCL28 = 28
  RSV_VCL29 = 29
  RSV_VCL30 = 30
  RSV_VCL31 = 31
  VPS_NUT = 32        # RBSP structure: video_parameter_set_rbsp( )
  SPS_NUT = 33        # RBSP structure: seq_parameter_set_rbsp( )
  PPS_NUT = 34        # RBSP structure: pic_parameter_set_rbsp( )
  AUD_NUT = 35
  EOS_NUT = 36
  EOB_NUT = 37
  FD_NUT = 38
  PREFIX_SEI_NUT = 39
  SUFFIX_SEI_NUT = 40
  RSV_NVCL41 = 41
  RSV_NVCL42 = 42
  RSV_NVCL43 = 43
  RSV_NVCL44 = 44
  RSV_NVCL45 = 45
  RSV_NVCL46 = 46
  RSV_NVCL47 = 47
  UNSPEC48 = 48
  UNSPEC49 = 49
  UNSPEC50 = 50
  UNSPEC51 = 51
  UNSPEC52 = 52
  UNSPEC53 = 53
  UNSPEC54 = 54
  UNSPEC55 = 55
  UNSPEC56 = 56
  UNSPEC57 = 57
  UNSPEC58 = 58
  UNSPEC59 = 59
  UNSPEC60 = 60
  UNSPEC61 = 61
  UNSPEC62 = 62
  UNSPEC63 = 63

# B.2.2 Byte stream NAL unit semantics
# - The nal_unit_type within the nal_unit( ) syntax structure is equal to VPS_NUT, SPS_NUT or PPS_NUT.
# - The byte stream NAL unit syntax structure contains the first NAL unit of an access unit in decoding
#   order, as specified in clause 7.4.2.4.4.
HEVC_PARAMETER_SET_NAL_UNITS = (
  HevcNalUnitType.VPS_NUT,
  HevcNalUnitType.SPS_NUT,
  HevcNalUnitType.PPS_NUT,
)

# 3.29 coded slice segment NAL unit: A NAL unit that has nal_unit_type in the range of TRAIL_N to RASL_R,
# inclusive, or in the range of BLA_W_LP to RSV_IRAP_VCL23, inclusive, which indicates that the NAL unit
# contains a coded slice segment
HEVC_CODED_SLICE_SEGMENT_NAL_UNITS = (
  HevcNalUnitType.TRAIL_N,
  HevcNalUnitType.TRAIL_R,
  HevcNalUnitType.TSA_N,
  HevcNalUnitType.TSA_R,
  HevcNalUnitType.STSA_N,
  HevcNalUnitType.STSA_R,
  HevcNalUnitType.RADL_N,
  HevcNalUnitType.RADL_R,
  HevcNalUnitType.RASL_N,
  HevcNalUnitType.RASL_R,
  HevcNalUnitType.BLA_W_LP,
  HevcNalUnitType.BLA_W_RADL,
  HevcNalUnitType.BLA_N_LP,
  HevcNalUnitType.IDR_W_RADL,
  HevcNalUnitType.IDR_N_LP,
  HevcNalUnitType.CRA_NUT,
)

class VideoFileInvalid(Exception):
  pass

def get_ue(dat: bytes, start_idx: int, skip_bits: int) -> tuple[int, int]:
  prefix_val = 0
  prefix_len = 0
  suffix_val = 0
  suffix_len = 0

  i = start_idx
  while i < len(dat):
    j = 7
    while j >= 0:
      if skip_bits > 0:
        skip_bits -= 1
      elif prefix_val == 0:
        prefix_val = (dat[i] >> j) & 1
        prefix_len += 1
      else:
        suffix_val = (suffix_val << 1) | ((dat[i] >> j) & 1)
        suffix_len += 1
      j -= 1

      if prefix_val == 1 and prefix_len - 1 == suffix_len:
        val = 2**(prefix_len-1) - 1 + suffix_val
        size = prefix_len + suffix_len
        return val, size
    i += 1

  raise VideoFileInvalid("invalid exponential-golomb code")

def require_nal_unit_start(dat: bytes, nal_unit_start: int) -> None:
  if nal_unit_start < 1:
    raise ValueError("start index must be greater than zero")

  if dat[nal_unit_start:nal_unit_start + NAL_UNIT_START_CODE_SIZE] != NAL_UNIT_START_CODE:
    raise VideoFileInvalid("data must begin with start code")

def get_hevc_nal_unit_length(dat: bytes, nal_unit_start: int) -> int:
  try:
    pos = dat.index(NAL_UNIT_START_CODE, nal_unit_start + NAL_UNIT_START_CODE_SIZE)
  except ValueError:
    pos = -1

  # length of NAL unit is byte count up to next NAL unit start index
  nal_unit_len = (pos if pos != -1 else len(dat)) - nal_unit_start
  if DEBUG:
    print("  nal_unit_len:", nal_unit_len)
  return nal_unit_len

def get_hevc_nal_unit_type(dat: bytes, nal_unit_start: int) -> HevcNalUnitType:
  # 7.3.1.2 NAL unit header syntax
  # nal_unit_header( ) {    // descriptor
  #   forbidden_zero_bit    f(1)
  #   nal_unit_type         u(6)
  #   nuh_layer_id          u(6)
  #   nuh_temporal_id_plus1 u(3)
  # }
  header_start = nal_unit_start + NAL_UNIT_START_CODE_SIZE
  nal_unit_header = dat[header_start:header_start + NAL_UNIT_HEADER_SIZE]
  if len(nal_unit_header) != 2:
    raise VideoFileInvalid("data to short to contain nal unit header")
  nal_unit_type = HevcNalUnitType((nal_unit_header[0] >> 1) & 0x3F)
  if DEBUG:
    print("  nal_unit_type:", nal_unit_type.name, f"({nal_unit_type.value})")
  return nal_unit_type

def get_hevc_slice_type(dat: bytes, nal_unit_start: int, nal_unit_type: HevcNalUnitType) -> tuple[int, bool]:
  # 7.3.2.9 Slice segment layer RBSP syntax
  # slice_segment_layer_rbsp( ) {
  #   slice_segment_header( )
  #   slice_segment_data( )
  #   rbsp_slice_segment_trailing_bits( )
  # }
  # ...
  # 7.3.6.1 General slice segment header syntax
  # slice_segment_header( ) {                                             // descriptor
  #   first_slice_segment_in_pic_flag                                     u(1)
  #   if( nal_unit_type >= BLA_W_LP && nal_unit_type <= RSV_IRAP_VCL23 )
  #     no_output_of_prior_pics_flag                                      u(1)
  #   slice_pic_parameter_set_id                                         ue(v)
  #   if( !first_slice_segment_in_pic_flag ) {
  #     if( dependent_slice_segments_enabled_flag )
  #       dependent_slice_segment_flag                                    u(1)
  #     slice_segment_address                                             u(v)
  #   }
  #   if( !dependent_slice_segment_flag ) {
  #     for( i = 0; i < num_extra_slice_header_bits; i++ )
  #       slice_reserved_flag[ i ]                                        u(1)
  #     slice_type                                                       ue(v)
  # ...

  rbsp_start = nal_unit_start + NAL_UNIT_START_CODE_SIZE + NAL_UNIT_HEADER_SIZE
  skip_bits = 0

  # 7.4.7.1 General slice segment header semantics
  # first_slice_segment_in_pic_flag equal to 1 specifies that the slice segment is the first slice segment of the picture in
  # decoding order. first_slice_segment_in_pic_flag equal to 0 specifies that the slice segment is not the first slice segment
  # of the picture in decoding order.
  is_first_slice = dat[rbsp_start] >> 7 & 1 == 1
  if not is_first_slice:
    # TODO: parse dependent_slice_segment_flag and slice_segment_address and get real slice_type
    # for now since we don't use it return -1 for slice_type
    return (-1, is_first_slice)
  skip_bits += 1 # skip past first_slice_segment_in_pic_flag

  if nal_unit_type >= HevcNalUnitType.BLA_W_LP and nal_unit_type <= HevcNalUnitType.RSV_IRAP_VCL23:
    # 7.4.7.1 General slice segment header semantics
    # no_output_of_prior_pics_flag affects the output of previously-decoded pictures in the decoded picture buffer after the
    # decoding of an IDR or a BLA picture that is not the first picture in the bitstream as specified in Annex C.
    skip_bits += 1 # skip past no_output_of_prior_pics_flag

  # 7.4.7.1 General slice segment header semantics
  # slice_pic_parameter_set_id specifies the value of pps_pic_parameter_set_id for the PPS in use.
  # The value of slice_pic_parameter_set_id shall be in the range of 0 to 63, inclusive.
  _, size = get_ue(dat, rbsp_start, skip_bits)
  skip_bits += size # skip past slice_pic_parameter_set_id

  # 7.4.3.3.1 General picture parameter set RBSP semanal_unit_lenntics
  # num_extra_slice_header_bits specifies the number of extra slice header bits that are present in the slice header RBSP
  # for coded pictures referring to the PPS. The value of num_extra_slice_header_bits shall be in the range of 0 to 2, inclusive,
  # in bitstreams conforming to this version of this Specification. Other values for num_extra_slice_header_bits are reserved
  # for future use by ITU-T | ISO/IEC. However, decoders shall allow num_extra_slice_header_bits to have any value.
  # TODO: get from PPS_NUT pic_parameter_set_rbsp( ) for corresponding slice_pic_parameter_set_id
  num_extra_slice_header_bits = 0
  skip_bits += num_extra_slice_header_bits

  # 7.4.7.1 General slice segment header semantics
  # slice_type specifies the coding type of the slice according to Table 7-7.
  # Table 7-7 - Name association to slice_type
  # slice_type | Name of slice_type
  #     0      | B (B slice)
  #     1      | P (P slice)
  #     2      | I (I slice)
  # unsigned integer 0-th order Exp-Golomb-coded syntax element with the left bit first
  slice_type, _ = get_ue(dat, rbsp_start, skip_bits)
  if DEBUG:
    print("  slice_type:", slice_type, f"(first slice: {is_first_slice})")
  if slice_type > 2:
    raise VideoFileInvalid("slice_type must be 0, 1, or 2")
  return slice_type, is_first_slice

def hevc_index(hevc_file_name: str, allow_corrupt: bool=False) -> tuple[list, int, bytes]:
  with FileReader(hevc_file_name) as f:
    dat = f.read()

  if len(dat) < NAL_UNIT_START_CODE_SIZE + 1:
    raise VideoFileInvalid("data is too short")

  if dat[0] != 0x00:
    raise VideoFileInvalid("first byte must be 0x00")

  prefix_dat = b""
  frame_types = list()

  i = 1 # skip past first byte 0x00
  try:
    while i < len(dat):
      require_nal_unit_start(dat, i)
      nal_unit_len = get_hevc_nal_unit_length(dat, i)
      nal_unit_type = get_hevc_nal_unit_type(dat, i)
      if nal_unit_type in HEVC_PARAMETER_SET_NAL_UNITS:
        prefix_dat += dat[i:i+nal_unit_len]
      elif nal_unit_type in HEVC_CODED_SLICE_SEGMENT_NAL_UNITS:
        slice_type, is_first_slice = get_hevc_slice_type(dat, i, nal_unit_type)
        if is_first_slice:
          frame_types.append((slice_type, i))
      i += nal_unit_len
  except Exception as e:
    if not allow_corrupt:
      raise
    print(f"ERROR: NAL unit skipped @ {i}\n", str(e))

  return frame_types, len(dat), prefix_dat

def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("input_file", type=str)
  parser.add_argument("output_prefix_file", type=str)
  parser.add_argument("output_index_file", type=str)
  args = parser.parse_args()

  frame_types, dat_len, prefix_dat = hevc_index(args.input_file)
  with open(args.output_prefix_file, "wb") as f:
    f.write(prefix_dat)

  with open(args.output_index_file, "wb") as f:
    for ft, fp in frame_types:
      f.write(struct.pack("<II", ft, fp))
    f.write(struct.pack("<II", 0xFFFFFFFF, dat_len))

if __name__ == "__main__":
  main()
