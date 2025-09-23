#include "registers_declarations.h"

static reg register_map[REGISTER_MAP_SIZE];

// Hash spread in first and second iterations seems to be reasonable.
// See: tests/development/register_hashmap_spread.py
// Also, check the collision warnings in the debug output, and minimize those.
static uint16_t hash_addr(uint32_t input){
  return (((input >> 16U) ^ ((((input + 1U) & 0xFFFFU) * HASHING_PRIME) & 0xFFFFU)) & REGISTER_MAP_SIZE);
}

// Do not put bits in the check mask that get changed by the hardware
void register_set(volatile uint32_t *addr, uint32_t val, uint32_t mask){
  ENTER_CRITICAL()
  // Set bits in register that are also in the mask
  (*addr) = ((*addr) & (~mask)) | (val & mask);

  // Add these values to the map
  uint16_t hash = hash_addr((uint32_t) addr);
  uint16_t tries = REGISTER_MAP_SIZE;
  while(CHECK_COLLISION(hash, addr) && (tries > 0U)) { hash = hash_addr((uint32_t) hash); tries--;}
  if (tries != 0U){
    register_map[hash].address = addr;
    register_map[hash].value = (register_map[hash].value & (~mask)) | (val & mask);
    register_map[hash].check_mask |= mask;
  } else {
    #ifdef DEBUG_FAULTS
      print("Hash collision: address 0x"); puth((uint32_t) addr); print("!\n");
    #endif
  }
  EXIT_CRITICAL()
}

// Set individual bits. Also add them to the check_mask.
// Do not use this to change bits that get reset by the hardware
void register_set_bits(volatile uint32_t *addr, uint32_t val) {
  register_set(addr, val, val);
}

// Clear individual bits. Also add them to the check_mask.
// Do not use this to clear bits that get set by the hardware
void register_clear_bits(volatile uint32_t *addr, uint32_t val) {
  register_set(addr, (~val), val);
}

// To be called periodically
void check_registers(void){
  for(uint16_t i=0U; i<REGISTER_MAP_SIZE; i++){
    if((uint32_t) register_map[i].address != 0U){
      ENTER_CRITICAL()
      if((*(register_map[i].address) & register_map[i].check_mask) != (register_map[i].value & register_map[i].check_mask)){
        if(!register_map[i].logged_fault){
          print("Register 0x"); puth((uint32_t) register_map[i].address); print(" divergent! Map: 0x"); puth(register_map[i].value); print(" Reg: 0x"); puth(*(register_map[i].address)); print("\n");
          register_map[i].logged_fault = true;
        }
        fault_occurred(FAULT_REGISTER_DIVERGENT);
      }
      EXIT_CRITICAL()
    }
  }
}

void init_registers(void) {
  for(uint16_t i=0U; i<REGISTER_MAP_SIZE; i++){
    register_map[i].address = (volatile uint32_t *) 0U;
    register_map[i].check_mask = 0U;
  }
}
