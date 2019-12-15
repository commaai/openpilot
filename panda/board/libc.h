// **** libc ****

void delay(int a) {
  volatile int i;
  for (i = 0; i < a; i++);
}

void *memset(void *str, int c, unsigned int n) {
  uint8_t *s = str;
  for (unsigned int i = 0; i < n; i++) {
    *s = c;
    s++;
  }
  return str;
}

void *memcpy(void *dest, const void *src, unsigned int n) {
  uint8_t *d = dest;
  const uint8_t *s = src;
  for (unsigned int i = 0; i < n; i++) {
    *d = *s;
    d++;
    s++;
  }
  return dest;
}

int memcmp(const void * ptr1, const void * ptr2, unsigned int num) {
  int ret = 0;
  const uint8_t *p1 = ptr1;
  const uint8_t *p2 = ptr2;
  for (unsigned int i = 0; i < num; i++) {
    if (*p1 != *p2) {
      ret = -1;
      break;
    }
    p1++;
    p2++;
  }
  return ret;
}

// ********************* IRQ helpers *********************

volatile bool interrupts_enabled = false;

void enable_interrupts(void) {
  interrupts_enabled = true;
  __enable_irq();
}

void disable_interrupts(void) {
  interrupts_enabled = false;
  __disable_irq();
}

uint8_t global_critical_depth = 0U;
#define ENTER_CRITICAL()                                      \
  __disable_irq();                                            \
  global_critical_depth += 1U;

#define EXIT_CRITICAL()                                       \
  global_critical_depth -= 1U;                                \
  if ((global_critical_depth == 0U) && interrupts_enabled) {  \
    __enable_irq();                                           \
  }

