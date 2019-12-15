uint8_t car_harness_status = 0U;
#define HARNESS_STATUS_NC 0U
#define HARNESS_STATUS_NORMAL 1U
#define HARNESS_STATUS_FLIPPED 2U

// Threshold voltage (mV) for either of the SBUs to be below before deciding harness is connected
#define HARNESS_CONNECTED_THRESHOLD 2500U

struct harness_configuration {
  const bool has_harness;
  GPIO_TypeDef *GPIO_SBU1;  
  GPIO_TypeDef *GPIO_SBU2;
  GPIO_TypeDef *GPIO_relay_normal;
  GPIO_TypeDef *GPIO_relay_flipped;
  uint8_t pin_SBU1;
  uint8_t pin_SBU2;
  uint8_t pin_relay_normal;
  uint8_t pin_relay_flipped;
  uint8_t adc_channel_SBU1;
  uint8_t adc_channel_SBU2;
};

// this function will be the API for tici
void set_intercept_relay(bool intercept) {
  if (car_harness_status != HARNESS_STATUS_NC) {
    if (intercept) {
      puts("switching harness to intercept (relay on)\n");
    } else {
      puts("switching harness to passthrough (relay off)\n");
    }

    if(car_harness_status == HARNESS_STATUS_NORMAL){
      set_gpio_output(current_board->harness_config->GPIO_relay_normal, current_board->harness_config->pin_relay_normal, !intercept);
    } else {
      set_gpio_output(current_board->harness_config->GPIO_relay_flipped, current_board->harness_config->pin_relay_flipped, !intercept);
    }
  }
}

bool harness_check_ignition(void) {
  bool ret = false;
  switch(car_harness_status){
    case HARNESS_STATUS_NORMAL:
      ret = !get_gpio_input(current_board->harness_config->GPIO_SBU2, current_board->harness_config->pin_SBU2);
      break;
    case HARNESS_STATUS_FLIPPED:
      ret = !get_gpio_input(current_board->harness_config->GPIO_SBU1, current_board->harness_config->pin_SBU1);
      break;
    default:
      break;
  }
  return ret;
}

// TODO: refactor to use harness config
void harness_setup_ignition_interrupts(void){
  if(car_harness_status == HARNESS_STATUS_NORMAL){
    SYSCFG->EXTICR[0] = SYSCFG_EXTICR1_EXTI3_PC;
    EXTI->IMR |= (1U << 3);
    EXTI->RTSR |= (1U << 3);
    EXTI->FTSR |= (1U << 3);
    puts("setup interrupts: normal\n");
  } else if(car_harness_status == HARNESS_STATUS_FLIPPED) {
    SYSCFG->EXTICR[0] = SYSCFG_EXTICR1_EXTI0_PC;
    EXTI->IMR |= (1U << 0);
    EXTI->RTSR |= (1U << 0);
    EXTI->FTSR |= (1U << 0);
    NVIC_EnableIRQ(EXTI1_IRQn);
    puts("setup interrupts: flipped\n");
  } else {
    puts("tried to setup ignition interrupts without harness connected\n");
  }
  NVIC_EnableIRQ(EXTI0_IRQn);
  NVIC_EnableIRQ(EXTI3_IRQn);
}

uint8_t harness_detect_orientation(void) {
  uint8_t ret = HARNESS_STATUS_NC;

  #ifndef BOOTSTUB
  uint32_t sbu1_voltage = adc_get(current_board->harness_config->adc_channel_SBU1);
  uint32_t sbu2_voltage = adc_get(current_board->harness_config->adc_channel_SBU2);

  // Detect connection and orientation
  if((sbu1_voltage < HARNESS_CONNECTED_THRESHOLD) || (sbu2_voltage < HARNESS_CONNECTED_THRESHOLD)){
    if (sbu1_voltage < sbu2_voltage) {
      // orientation normal
      ret = HARNESS_STATUS_NORMAL;
    } else {
      // orientation flipped
      ret = HARNESS_STATUS_FLIPPED;
    }
  }
  #endif

  return ret;
}

void harness_init(void) {
  // delay such that the connection is fully made before trying orientation detection
  current_board->set_led(LED_BLUE, true);
  delay(10000000);
  current_board->set_led(LED_BLUE, false);

  // try to detect orientation
  uint8_t ret = harness_detect_orientation();
  if (ret != HARNESS_STATUS_NC) {
    puts("detected car harness with orientation "); puth2(ret); puts("\n");
    car_harness_status = ret;

    // set the SBU lines to be inputs before using the relay. The lines are not 5V tolerant in ADC mode!
    set_gpio_mode(current_board->harness_config->GPIO_SBU1, current_board->harness_config->pin_SBU1, MODE_INPUT);
    set_gpio_mode(current_board->harness_config->GPIO_SBU2, current_board->harness_config->pin_SBU2, MODE_INPUT);

    // now we have orientation, set pin ignition detection
    if(car_harness_status == HARNESS_STATUS_NORMAL){
      set_gpio_mode(current_board->harness_config->GPIO_SBU2, current_board->harness_config->pin_SBU2, MODE_INPUT);
    } else {
      set_gpio_mode(current_board->harness_config->GPIO_SBU1, current_board->harness_config->pin_SBU1, MODE_INPUT);
    }      

    // keep busses connected by default
    set_intercept_relay(false);

    // setup ignition interrupts
    harness_setup_ignition_interrupts();
  } else {
    puts("failed to detect car harness!\n");
  }
}