// Early bringup
extern void *g_pfnVectors;
extern uint32_t enter_bootloader_mode;

void early_initialization(void) {
  SystemInit();
  // after it's been in the bootloader, things are initted differently, so we reset
  if ((enter_bootloader_mode != BOOT_NORMAL) &&
      (enter_bootloader_mode != ENTER_SOFTLOADER_MAGIC)) {
    enter_bootloader_mode = BOOT_NORMAL;
    NVIC_SystemReset();
  }
  // setup interrupt table
  SCB->VTOR = (uint32_t)&g_pfnVectors; // TODO: check if SystemInit is enough!
}
