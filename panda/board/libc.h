// **** libc ****

void delay(int a) {
  volatile int i;
  for (i = 0; i < a; i++);
}

void *memset(void *str, int c, unsigned int n) {
  unsigned int i;
  for (i = 0; i < n; i++) {
    *((uint8_t*)str) = c;
    ++str;
  }
  return str;
}

void *memcpy(void *dest, const void *src, unsigned int n) {
  unsigned int i;
  // TODO: make not slow
  for (i = 0; i < n; i++) {
    ((uint8_t*)dest)[i] = *(uint8_t*)src;
    ++src;
  }
  return dest;
}

int memcmp(const void * ptr1, const void * ptr2, unsigned int num) {
  unsigned int i;
  int ret = 0;
  for (i = 0; i < num; i++) {
    if ( ((uint8_t*)ptr1)[i] != ((uint8_t*)ptr2)[i] ) {
      ret = -1;
      break;
    }
  }
  return ret;
}

// ********************* IRQ helpers *********************

int interrupts_enabled = 0;
void enable_interrupts(void) {
  interrupts_enabled = 1;
  __enable_irq();
}

int critical_depth = 0;
void enter_critical_section(void) {
  __disable_irq();
  // this is safe because interrupts are disabled
  critical_depth += 1;
}

void exit_critical_section(void) {
  // this is safe because interrupts are disabled
  critical_depth -= 1;
  if ((critical_depth == 0) && interrupts_enabled) {
    __enable_irq();
  }
}

