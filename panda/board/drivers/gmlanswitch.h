#define GMLAN_TICKS_PER_SECOND 33300 //1sec @ 33.3kbps
#define GMLAN_TICKS_PER_TIMEOUT_TICKLE 500 //15ms @ 33.3kbps
#define GMLAN_HIGH 0 //0 is high on bus (dominant)
#define GMLAN_LOW 1 //1 is low on bus

#ifdef PANDA
int gmlan_timeout_counter = GMLAN_TICKS_PER_TIMEOUT_TICKLE; //GMLAN transceiver times out every 17ms held high; tickle every 15ms
int can_timeout_counter = GMLAN_TICKS_PER_SECOND; //1 second

int inverted_bit_to_send = GMLAN_HIGH; 
int gmlan_switch_enabled = -1;

void TIM4_IRQHandler(void) {
  if (TIM4->SR & TIM_SR_UIF && gmlan_switch_enabled != -1) {
    if (can_timeout_counter == 0) {
      //it has been more than 1 second since receiving a CAN message in tesla safety. Assume we are in a different safety mode, disable timer and restore the GMLAN output
      set_gpio_output(GPIOB, 13, GMLAN_LOW);
      //set_gpio_mode(GPIOB, 13, MODE_INPUT);
      //TIM4->DIER = 0;  // no update interrupt
      //TIM4->CR1 = 0;   // disable timer
      gmlan_switch_enabled = -1;
    }
    else {
      can_timeout_counter--;
      if (gmlan_timeout_counter == 0) {
        //Send a 1 (bus low) every 15ms to reset the GMLAN transceivers timeout
        gmlan_timeout_counter = GMLAN_TICKS_PER_TIMEOUT_TICKLE;
        set_gpio_output(GPIOB, 13, GMLAN_LOW);
      }
      else {
        set_gpio_output(GPIOB, 13, inverted_bit_to_send);
        gmlan_timeout_counter--;
      }
    }
  }
  TIM4->SR = 0;
}

void gmlan_switch_init(void) {
  gmlan_switch_enabled = 1;
  set_gpio_mode(GPIOB, 13, MODE_OUTPUT);
  
  // setup
  TIM4->PSC = 48-1;          // tick on 1 us
  TIM4->CR1 = TIM_CR1_CEN;   // enable
  TIM4->ARR = 30-1;          // 33.3 kbps

  // in case it's disabled
  NVIC_EnableIRQ(TIM4_IRQn);

  // run the interrupt
  TIM4->DIER = TIM_DIER_UIE; // update interrupt
  TIM4->SR = 0;
  
  inverted_bit_to_send = GMLAN_HIGH; //We got initialized, set the output high 
}

void set_gmlan_digital_output(int to_set) {
  inverted_bit_to_send = to_set;
  /*
  puts("Writing ");
  puth(inverted_bit_to_send);
  puts("\n");
  */
}

void enable_gmlan_switch(void) {
  can_timeout_counter = GMLAN_TICKS_PER_SECOND;
  gmlan_switch_enabled = 1; 
  inverted_bit_to_send = GMLAN_HIGH;
}

#endif