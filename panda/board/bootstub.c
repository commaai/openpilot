#define BOOTSTUB

#include "config.h"
#include "obj/gitversion.h"

#ifdef STM32F4
  #define PANDA
  #include "stm32f4xx.h"
  #include "stm32f4xx_hal_gpio_ex.h"
#else
  #include "stm32f2xx.h"
  #include "stm32f2xx_hal_gpio_ex.h"
#endif

// default since there's no serial
void puts(const char *a) {}
void puth(unsigned int i) {}

#include "libc.h"
#include "provision.h"

#include "drivers/clock.h"
#include "drivers/llgpio.h"
#include "gpio.h"

#include "drivers/spi.h"
#include "drivers/usb.h"
//#include "drivers/uart.h"

#include "crypto/rsa.h"
#include "crypto/sha.h"

#include "obj/cert.h"

#include "spi_flasher.h"

void __initialize_hardware_early() {
  early();
}

void fail() {
  soft_flasher_start();
}

// know where to sig check
extern void *_app_start[];

// FIXME: sometimes your panda will fail flashing and will quickly blink a single Green LED
// BOUNTY: $200 coupon on shop.comma.ai or $100 check.

int main() {
  __disable_irq();
  clock_init();
  detect();

  if (revision == PANDA_REV_C) {
    set_usb_power_mode(USB_POWER_CLIENT);
  }

  if (enter_bootloader_mode == ENTER_SOFTLOADER_MAGIC) {
    enter_bootloader_mode = 0;
    soft_flasher_start();
  }

  // validate length
  int len = (int)_app_start[0];
  if ((len < 8) || (len > (0x1000000 - 0x4000 - 4 - RSANUMBYTES))) goto fail;

  // compute SHA hash
  uint8_t digest[SHA_DIGEST_SIZE];
  SHA_hash(&_app_start[1], len-4, digest);

  // verify RSA signature
  if (RSA_verify(&release_rsa_key, ((void*)&_app_start[0]) + len, RSANUMBYTES, digest, SHA_DIGEST_SIZE)) {
    goto good;
  }

  // allow debug if built from source
#ifdef ALLOW_DEBUG
  if (RSA_verify(&debug_rsa_key, ((void*)&_app_start[0]) + len, RSANUMBYTES, digest, SHA_DIGEST_SIZE)) {
    goto good;
  }
#endif

// here is a failure
fail:
  fail();
  return 0;
good:
  // jump to flash
  ((void(*)()) _app_start[1])();
  return 0;
}

