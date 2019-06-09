#define POWER_SAVE_STATUS_DISABLED 0
#define POWER_SAVE_STATUS_ENABLED 1

int power_save_status = POWER_SAVE_STATUS_DISABLED;

void power_save_enable(void) {
  if (power_save_status == POWER_SAVE_STATUS_ENABLED) return;
  puts("enable power savings\n");

  // turn off can
  set_can_enable(CAN1, 0);
  set_can_enable(CAN2, 0);
  set_can_enable(CAN3, 0);

  // turn off GMLAN
  set_gpio_output(GPIOB, 14, 0);
  set_gpio_output(GPIOB, 15, 0);

  // turn off LIN
  set_gpio_output(GPIOB, 7, 0);
  set_gpio_output(GPIOA, 14, 0);

  if (is_grey_panda) {
    char UBLOX_SLEEP_MSG[] = "\xb5\x62\x06\x04\x04\x00\x01\x00\x08\x00\x17\x78";
    uart_ring *ur = get_ring_by_number(1);
    for (int i = 0; i < sizeof(UBLOX_SLEEP_MSG)-1; i++) while (!putc(ur, UBLOX_SLEEP_MSG[i]));
  }

  power_save_status = POWER_SAVE_STATUS_ENABLED;
}

void power_save_disable(void) {
  if (power_save_status == POWER_SAVE_STATUS_DISABLED) return;
  puts("disable power savings\n");

  // turn on can
  set_can_enable(CAN1, 1);
  set_can_enable(CAN2, 1);
  set_can_enable(CAN3, 1);

  // turn on GMLAN
  set_gpio_output(GPIOB, 14, 1);
  set_gpio_output(GPIOB, 15, 1);

  // turn on LIN
  set_gpio_output(GPIOB, 7, 1);
  set_gpio_output(GPIOA, 14, 1);

  if (is_grey_panda) {
    char UBLOX_WAKE_MSG[] = "\xb5\x62\x06\x04\x04\x00\x01\x00\x09\x00\x18\x7a";
    uart_ring *ur = get_ring_by_number(1);
    for (int i = 0; i < sizeof(UBLOX_WAKE_MSG)-1; i++) while (!putc(ur, UBLOX_WAKE_MSG[i]));
  }

  power_save_status = POWER_SAVE_STATUS_DISABLED;
}

