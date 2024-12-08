#pragma once

typedef struct reg {
  volatile uint32_t *address;
  uint32_t value;
  uint32_t check_mask;
} reg;

// 10 bit hash with 23 as a prime
#define REGISTER_MAP_SIZE 0x3FFU
#define HASHING_PRIME 23U
#define CHECK_COLLISION(hash, addr) (((uint32_t) register_map[hash].address != 0U) && (register_map[hash].address != (addr)))

// Do not put bits in the check mask that get changed by the hardware
void register_set(volatile uint32_t *addr, uint32_t val, uint32_t mask);
// Set individual bits. Also add them to the check_mask.
// Do not use this to change bits that get reset by the hardware
void register_set_bits(volatile uint32_t *addr, uint32_t val);
// Clear individual bits. Also add them to the check_mask.
// Do not use this to clear bits that get set by the hardware
void register_clear_bits(volatile uint32_t *addr, uint32_t val);
// To be called periodically
void check_registers(void);
void init_registers(void);
