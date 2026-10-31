#pragma once

// Early bringup
#define ENTER_BOOTLOADER_MAGIC 0xdeadbeefU
#define ENTER_SOFTLOADER_MAGIC 0xdeadc0deU
#define BOOT_NORMAL 0xdeadb111U

extern void *g_pfnVectors;
extern uint32_t enter_bootloader_mode;

typedef void (*bootloader_fcn)(void);
typedef bootloader_fcn *bootloader_fcn_ptr;

static void jump_to_bootloader(void) {
  // do enter bootloader
  enter_bootloader_mode = 0;

  bootloader_fcn_ptr bootloader_ptr = (bootloader_fcn_ptr)BOOTLOADER_ADDRESS;
  bootloader_fcn bootloader = *bootloader_ptr;

  // jump to bootloader
  enable_interrupts();
  bootloader();

  // reset on exit
  enter_bootloader_mode = BOOT_NORMAL;
  NVIC_SystemReset();
}

void early_initialization(void) {
  // Reset global critical depth
  disable_interrupts();
  global_critical_depth = 0;

  // Init register and interrupt tables
  init_registers();

  // after it's been in the bootloader, things are initted differently, so we reset
  if ((enter_bootloader_mode != BOOT_NORMAL) &&
      (enter_bootloader_mode != ENTER_BOOTLOADER_MAGIC) &&
      (enter_bootloader_mode != ENTER_SOFTLOADER_MAGIC)) {
    enter_bootloader_mode = BOOT_NORMAL;
    NVIC_SystemReset();
  }

  // if wrong chip, reboot
  volatile unsigned int id = DBGMCU->IDCODE;
  if ((id & 0xFFFU) != MCU_IDCODE) {
    enter_bootloader_mode = ENTER_BOOTLOADER_MAGIC;
  }

  // setup interrupt table
  register_set(&(SCB->VTOR), (uint32_t)&g_pfnVectors, SCB_VTOR_TBLOFF_Msk);

  // early GPIOs float everything
  early_gpio_float();

  detect_board_type();

  if (enter_bootloader_mode == ENTER_BOOTLOADER_MAGIC) {
    led_init();
    led_set(LED_GREEN, 1);
    jump_to_bootloader();
  }
}

// we might hit NMI / HardFault close to brown-out.
// we don't want to infinitely loop, so let's set a logging cookie and reset.
#define NMI_COOKIE 0x4E4D4921U
#define HARDFAULT_COOKIE 0x48464C54U
extern uint32_t fault_cookie;
extern uint32_t fault_cfsr;
extern uint32_t fault_addr;
extern uint32_t fault_pc;

// frame is the stacked r0-r3, r12, lr, pc and xpsr, panda only runs on the MSP
static void fault_record(uint32_t cookie, const uint32_t *frame) {
  fault_cookie = cookie;
  if (cookie == HARDFAULT_COOKIE) {
    fault_cfsr = SCB->CFSR;
    fault_addr = ((fault_cfsr & SCB_CFSR_MMARVALID_Msk) != 0U) ? SCB->MMFAR : SCB->BFAR;
  }
  // a failed stacking leaves no frame to read
  fault_pc = ((SCB->CFSR & (SCB_CFSR_MSTKERR_Msk | SCB_CFSR_STKERR_Msk)) == 0U) ? frame[6] : 0U;
  NVIC_SystemReset();
}

__attribute__((used)) static void nmi_record(const uint32_t *frame) {
  fault_record(NMI_COOKIE, frame);
}

__attribute__((used)) static void hardfault_record(const uint32_t *frame) {
  fault_record(HARDFAULT_COOKIE, frame);
}

// naked to read the frame before anything is pushed
__attribute__((naked)) void NMI_Handler(void) {
  // cppcheck-suppress misra-c2012-17.3 ; inline asm
  __asm volatile ("mrs r0, msp\n b nmi_record");
}

__attribute__((naked)) void HardFault_Handler(void) {
  // cppcheck-suppress misra-c2012-17.3 ; inline asm
  __asm volatile ("mrs r0, msp\n b hardfault_record");
}
