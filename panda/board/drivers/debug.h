#pragma once

#include "board/drivers/drivers.h"

#define DEBUG_BUFFER_SIZE 0x400U

static char debug_buffer[DEBUG_BUFFER_SIZE];
static volatile uint16_t debug_write_ptr = 0;
static volatile uint16_t debug_read_ptr = 0;

bool debug_get_char(char *elem) {
  bool ret = false;

  ENTER_CRITICAL();
  if (debug_write_ptr != debug_read_ptr) {
    if (elem != NULL) *elem = debug_buffer[debug_read_ptr];
    debug_read_ptr = (debug_read_ptr + 1U) % DEBUG_BUFFER_SIZE;
    ret = true;
  }
  EXIT_CRITICAL();

  return ret;
}

static void putch(const char a) {
  ENTER_CRITICAL();
  uint16_t next_write_ptr = (debug_write_ptr + 1U) % DEBUG_BUFFER_SIZE;
  if (next_write_ptr == debug_read_ptr) {
    // Drop the oldest byte when the buffer is full.
    debug_read_ptr = (debug_read_ptr + 1U) % DEBUG_BUFFER_SIZE;
  }
  debug_buffer[debug_write_ptr] = a;
  debug_write_ptr = next_write_ptr;
  EXIT_CRITICAL();
}

void print(const char *a) {
  for (const char *in = a; *in; in++) {
    if (*in == '\n') putch('\r');
    putch(*in);
  }
}

static void puthx(uint32_t i, uint8_t len) {
  const char c[] = "0123456789abcdef";
  for (int pos = ((int)len * 4) - 4; pos > -4; pos -= 4) {
    putch(c[(i >> (unsigned int)(pos)) & 0xFU]);
  }
}

void puth(unsigned int i) {
  puthx(i, 8U);
}

static void hexdump(const void *a, int l) {
  if (a != NULL) {
    for (int i=0; i < l; i++) {
      if ((i != 0) && ((i & 0xf) == 0)) print("\n");
      puthx(((const unsigned char*)a)[i], 2U);
      print(" ");
    }
  }
  print("\n");
}

static inline void puth4(unsigned int i) {
  puthx(i, 4U);
}
