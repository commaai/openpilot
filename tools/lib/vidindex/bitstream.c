#include "./bitstream.h"

#include <stdbool.h>
#include <assert.h>

static const uint32_t BS_MASKS[33] = {
    0,           0x1L,        0x3L,       0x7L,       0xFL,       0x1FL,
    0x3FL,       0x7FL,       0xFFL,      0x1FFL,     0x3FFL,     0x7FFL,
    0xFFFL,      0x1FFFL,     0x3FFFL,    0x7FFFL,    0xFFFFL,    0x1FFFFL,
    0x3FFFFL,    0x7FFFFL,    0xFFFFFL,   0x1FFFFFL,  0x3FFFFFL,  0x7FFFFFL,
    0xFFFFFFL,   0x1FFFFFFL,  0x3FFFFFFL, 0x7FFFFFFL, 0xFFFFFFFL, 0x1FFFFFFFL,
    0x3FFFFFFFL, 0x7FFFFFFFL, 0xFFFFFFFFL};

void bs_init(struct bitstream* bs, const uint8_t* buffer, size_t input_size) {
  bs->buffer_ptr = buffer;
  bs->buffer_end = buffer + input_size;
  bs->value = 0;
  bs->pos = 0;
  bs->shift = 8;
  bs->size = input_size * 8;
}

uint32_t bs_get(struct bitstream* bs, int n) {
  if (n > 32)
    return 0;

  bs->pos += n;
  bs->shift += n;
  while (bs->shift > 8) {
    if (bs->buffer_ptr < bs->buffer_end) {
      bs->value <<= 8;
      bs->value |= *bs->buffer_ptr++;
      bs->shift -= 8;
    } else {
      bs_seek(bs, bs->pos - n);
      return 0;
      // bs->value <<= 8;
      // bs->shift -= 8;
    }
  }
  return (bs->value >> (8 - bs->shift)) & BS_MASKS[n];
}

void bs_seek(struct bitstream* bs, size_t new_pos) {
  bs->pos = (new_pos / 32) * 32;
  bs->shift = 8;
  bs->value = 0;
  bs_get(bs, new_pos % 32);
}

uint32_t bs_peek(struct bitstream* bs, int n) {
  struct bitstream bak = *bs;
  return bs_get(&bak, n);
}

size_t bs_remain(struct bitstream* bs) {
  return bs->size - bs->pos;
}

int bs_eof(struct bitstream* bs) {
  return bs_remain(bs) == 0;
}

uint32_t bs_ue(struct bitstream* bs) {
  static const uint8_t exp_golomb_bits[256] = {
      8, 7, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3,
      3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
      2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };
  uint32_t bits, read = 0;
  int bits_left;
  uint8_t coded;
  int done = 0;
  bits = 0;
  // we want to read 8 bits at a time - if we don't have 8 bits,
  // read what's left, and shift.  The exp_golomb_bits calc remains the
  // same.
  while (!done) {
    bits_left = bs_remain(bs);
    if (bits_left < 8) {
      read = bs_peek(bs, bits_left) << (8 - bits_left);
      done = 1;
    } else {
      read = bs_peek(bs, 8);
      if (read == 0) {
        bs_get(bs, 8);
        bits += 8;
      } else {
        done = 1;
      }
    }
  }
  coded = exp_golomb_bits[read];
  bs_get(bs, coded);
  bits += coded;

  //  printf("ue - bits %d\n", bits);
  return bs_get(bs, bits + 1) - 1;
}

int32_t bs_se(struct bitstream* bs) {
  uint32_t ret;
  ret = bs_ue(bs);
  if ((ret & 0x1) == 0) {
    ret >>= 1;
    int32_t temp = 0 - ret;
    return temp;
  }
  return (ret + 1) >> 1;
}
