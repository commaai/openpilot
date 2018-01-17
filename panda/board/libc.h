// **** shitty libc ****

void delay(int a) {
  volatile int i;
  for (i=0;i<a;i++);
}

void *memset(void *str, int c, unsigned int n) {
  int i;
  for (i = 0; i < n; i++) {
    *((uint8_t*)str) = c;
    ++str;
  }
  return str;
}

void *memcpy(void *dest, const void *src, unsigned int n) {
  int i;
  // TODO: make not slow
  for (i = 0; i < n; i++) {
    ((uint8_t*)dest)[i] = *(uint8_t*)src;
    ++src;
  }
  return dest;
}

int memcmp(const void * ptr1, const void * ptr2, unsigned int num) {
  int i;
  for (i = 0; i < num; i++) {
    if ( ((uint8_t*)ptr1)[i] != ((uint8_t*)ptr2)[i] ) return -1;
  }
  return 0;
}

// ********************* IRQ helpers *********************

int critical_depth = 0;
void enter_critical_section() {
  __disable_irq();
  // this is safe because interrupts are disabled
  critical_depth += 1;
}

void exit_critical_section() {
  // this is safe because interrupts are disabled
  critical_depth -= 1;
  if (critical_depth == 0) {
    __enable_irq();
  }
}

