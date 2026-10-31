#pragma once

// linker symbols, see stm32h7x5_flash.ld
extern uint32_t _ssafety[16];
extern uint32_t _stack_guard;

#define MPU_REGION_SAFETY 0U
#define MPU_REGION_STACK_GUARD 1U

// read only once FAULT_MPU is permanent, until then a write from outside the safety code is caught in mpu_safety_open()
#define MPU_SAFETY_CLOSED (((PERMANENT_FAULTS & FAULT_MPU) != 0U) ? ARM_MPU_AP_PRO : ARM_MPU_AP_PRIV)

static uint8_t mpu_safety_depth = 0U;
static uint32_t mpu_safety_copy[16];
static bool mpu_safety_changed = false;
static uint32_t mpu_safety_changed_addr = 0U;

// normal memory, non-cacheable, never executed
static uint32_t mpu_rasr(uint32_t access, uint32_t size) {
  return MPU_RASR_XN_Msk | (access << MPU_RASR_AP_Pos) | (1UL << MPU_RASR_TEX_Pos) | (size << MPU_RASR_SIZE_Pos) | MPU_RASR_ENABLE_Msk;
}

static uint32_t mpu_safety_rasr(uint32_t access) {
  return mpu_rasr(access, ARM_MPU_REGION_SIZE_64B);
}

static uint32_t mpu_stack_guard_rasr(void) {
  return mpu_rasr(ARM_MPU_AP_NONE, ARM_MPU_REGION_SIZE_512B);
}

static void mpu_set_safety_access(uint32_t access) {
  __DMB();
  MPU->RNR = MPU_REGION_SAFETY;
  MPU->RASR = mpu_safety_rasr(access);
  __DSB();
  __ISB();
}

// only safety code writes the safety state, interrupts stay off while it's writable
void mpu_safety_open(void) {
  ENTER_CRITICAL();
  if (mpu_safety_depth == 0U) {
    // body firmware runs without the MPU
    if ((MPU->CTRL & MPU_CTRL_ENABLE_Msk) != 0U) {
      for (uint8_t i = 0U; i < 16U; i++) {
        if (!mpu_safety_changed && (_ssafety[i] != mpu_safety_copy[i])) {
          mpu_safety_changed = true;
          mpu_safety_changed_addr = (uint32_t)&_ssafety[i];
        }
      }
    }
    mpu_set_safety_access(ARM_MPU_AP_PRIV);
  }
  mpu_safety_depth += 1U;
}

void mpu_safety_close(void) {
  mpu_safety_depth -= 1U;
  if (mpu_safety_depth == 0U) {
    mpu_set_safety_access(MPU_SAFETY_CLOSED);
    (void)memcpy(mpu_safety_copy, _ssafety, sizeof(mpu_safety_copy));
  }
  EXIT_CRITICAL();
}

void mpu_init(void) {
  MPU->RNR = MPU_REGION_STACK_GUARD;
  MPU->RBAR = (uint32_t)&_stack_guard;
  MPU->RASR = mpu_stack_guard_rasr();

  MPU->RNR = MPU_REGION_SAFETY;
  MPU->RBAR = (uint32_t)&_ssafety;
  MPU->RASR = mpu_safety_rasr(MPU_SAFETY_CLOSED);
  (void)memcpy(mpu_safety_copy, _ssafety, sizeof(mpu_safety_copy));

  // privileged default map for everything else, a violation escalates to HardFault
  register_set(&(MPU->CTRL), MPU_CTRL_PRIVDEFENA_Msk | MPU_CTRL_ENABLE_Msk, 0x7U);
  __DSB();
  __ISB();
}

static bool mpu_region_ok(uint32_t region, uint32_t base, uint32_t rasr) {
  MPU->RNR = region;
  return ((MPU->RBAR & MPU_RBAR_ADDR_Msk) == base) && (MPU->RASR == rasr);
}

// RBAR and RASR are banked by RNR, so register_set can't read them back
void mpu_check(void) {
  ENTER_CRITICAL();
  uint32_t bad = mpu_region_ok(MPU_REGION_STACK_GUARD, (uint32_t)&_stack_guard, mpu_stack_guard_rasr()) ? 0U : (1UL << MPU_REGION_STACK_GUARD);
  bad |= mpu_region_ok(MPU_REGION_SAFETY, (uint32_t)&_ssafety, mpu_safety_rasr(MPU_SAFETY_CLOSED)) ? 0U : (1UL << MPU_REGION_SAFETY);
  for (uint32_t region = MPU_REGION_STACK_GUARD + 1U; region < ((MPU->TYPE & MPU_TYPE_DREGION_Msk) >> MPU_TYPE_DREGION_Pos); region++) {
    MPU->RNR = region;
    bad |= ((MPU->RASR & MPU_RASR_ENABLE_Msk) == 0U) ? 0U : (1UL << region);
  }
  EXIT_CRITICAL();
  if (bad != 0U) {
    fault_occurred_value(FAULT_REGISTER_DIVERGENT, bad);
  }
  if (mpu_safety_changed) {
    fault_occurred_value(FAULT_MPU, mpu_safety_changed_addr);
  }
}
