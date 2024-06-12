extern uint8_t hw_type;
board_t board;

void board_detect(void) {
  hw_type = board_id();
  // 0 = base, 3 = knee
  if (hw_type == HW_TYPE_BASE) {
    board.hall_left.hall_portA = GPIOC;
    board.hall_left.hall_pinA = GPIO_PIN_13;
    board.hall_left.hall_portB = GPIOC;
    board.hall_left.hall_pinB = GPIO_PIN_14;
    board.hall_left.hall_portC = GPIOC;
    board.hall_left.hall_pinC = GPIO_PIN_15;

    board.hall_right.hall_portA = GPIOC;
    board.hall_right.hall_pinA = GPIO_PIN_10;
    board.hall_right.hall_portB = GPIOC;
    board.hall_right.hall_pinB = GPIO_PIN_11;
    board.hall_right.hall_portC = GPIOC;
    board.hall_right.hall_pinC = GPIO_PIN_12;

    board.CAN = CAN2;
    board.can_alt_tx = GPIO_AF9_CAN2;
    board.can_alt_rx = GPIO_AF9_CAN2;
    board.can_pinRX = GPIO_PIN_5;
    board.can_portRX = GPIOB;
    board.can_pinTX = GPIO_PIN_6;
    board.can_portTX = GPIOB;
    board.can_pinEN = GPIO_PIN_7;
    board.can_portEN = GPIOB;

    board.ignition_pin = GPIO_PIN_9;
    board.ignition_port = GPIOB;

    board.led_pinR = GPIO_PIN_2;
    board.led_portR = GPIOD;
    board.led_pinG = GPIO_PIN_15;
    board.led_portG = GPIOA;
    board.led_pinB = GPIO_PIN_1;
    board.led_portB = GPIOC;

    board.can_addr_offset = 0x0U;
    board.uds_offset = 0x0U;

  } else if (hw_type == HW_TYPE_KNEE) {
    board.hall_left.hall_portA = GPIOC;
    board.hall_left.hall_pinA = GPIO_PIN_14;
    board.hall_left.hall_portB = GPIOC;
    board.hall_left.hall_pinB = GPIO_PIN_15;
    board.hall_left.hall_portC = GPIOC;
    board.hall_left.hall_pinC = GPIO_PIN_13;

    board.hall_right.hall_portA = GPIOD;
    board.hall_right.hall_pinA = GPIO_PIN_2;
    board.hall_right.hall_portB = GPIOC;
    board.hall_right.hall_pinB = GPIO_PIN_0;
    board.hall_right.hall_portC = GPIOC;
    board.hall_right.hall_pinC = GPIO_PIN_1;

    board.CAN = CAN1;
    board.can_alt_tx = GPIO_AF8_CAN1;
    board.can_alt_rx = GPIO_AF9_CAN1;
    board.can_pinRX = GPIO_PIN_11;
    board.can_portRX = GPIOA;
    board.can_pinTX = GPIO_PIN_9;
    board.can_portTX = GPIOB;
    board.can_pinEN = 0; // No pin, pulled down with 10k resistor
    board.can_portEN = GPIOB;

    board.ignition_pin = 0; // No pin, always enabled
    board.ignition_port = GPIOB;

    board.led_pinR = GPIO_PIN_2;
    board.led_portR = GPIOB;
    board.led_pinG = GPIO_PIN_15;
    board.led_portG = GPIOA;
    board.led_pinB = GPIO_PIN_5;
    board.led_portB = GPIOB;

    board.can_addr_offset = KNEE_ADDR_OFFSET;
    board.uds_offset = 0x10U;

    #ifndef BOOTSTUB
    MX_I2C_Init();
    #endif
  } else {
    // Fail to detect, halt
    while(1) {}
  }
}
