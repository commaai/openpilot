#ifndef bitstream_H
#define bitstream_H


#include <stddef.h>
#include <stdint.h>

struct bitstream {
  const uint8_t *buffer_ptr;
  const uint8_t *buffer_end;
  uint64_t value;
  uint32_t pos;
  uint32_t shift;
  size_t size;
};

void bs_init(struct bitstream *bs, const uint8_t *buffer, size_t input_size);
void bs_seek(struct bitstream *bs, size_t new_pos);
uint32_t bs_get(struct bitstream *bs, int n);
uint32_t bs_peek(struct bitstream *bs, int n);
size_t bs_remain(struct bitstream *bs);
int bs_eof(struct bitstream *bs);
uint32_t bs_ue(struct bitstream *bs);
int32_t bs_se(struct bitstream *bs);

#endif
