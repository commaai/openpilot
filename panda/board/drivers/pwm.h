#define PWM_COUNTER_OVERFLOW 2000U // To get ~50kHz

// TODO: Implement for 32-bit timers

void pwm_init(TIM_TypeDef *TIM, uint8_t channel){
    // Enable timer and auto-reload
    register_set(&(TIM->CR1), TIM_CR1_CEN | TIM_CR1_ARPE, 0x3FU);

    // Set channel as PWM mode 1 and enable output
    switch(channel){
        case 1U:
            register_set_bits(&(TIM->CCMR1), (TIM_CCMR1_OC1M_2 | TIM_CCMR1_OC1M_1 | TIM_CCMR1_OC1PE));
            register_set_bits(&(TIM->CCER), TIM_CCER_CC1E);
            break;
        case 2U:
            register_set_bits(&(TIM->CCMR1), (TIM_CCMR1_OC2M_2 | TIM_CCMR1_OC2M_1 | TIM_CCMR1_OC2PE));
            register_set_bits(&(TIM->CCER), TIM_CCER_CC2E);
            break;
        case 3U:
            register_set_bits(&(TIM->CCMR2), (TIM_CCMR2_OC3M_2 | TIM_CCMR2_OC3M_1 | TIM_CCMR2_OC3PE));
            register_set_bits(&(TIM->CCER), TIM_CCER_CC3E);
            break;
        case 4U:
            register_set_bits(&(TIM->CCMR2), (TIM_CCMR2_OC4M_2 | TIM_CCMR2_OC4M_1 | TIM_CCMR2_OC4PE));
            register_set_bits(&(TIM->CCER), TIM_CCER_CC4E);
            break;
        default:
            break;
    }

    // Set max counter value
    register_set(&(TIM->ARR), PWM_COUNTER_OVERFLOW, 0xFFFFU);

    // Update registers and clear counter
    TIM->EGR |= TIM_EGR_UG;
}

void pwm_set(TIM_TypeDef *TIM, uint8_t channel, uint8_t percentage){
    uint16_t comp_value = (((uint16_t) percentage * PWM_COUNTER_OVERFLOW) / 100U);
    switch(channel){
        case 1U:
            register_set(&(TIM->CCR1), comp_value, 0xFFFFU);
            break;
        case 2U:
            register_set(&(TIM->CCR2), comp_value, 0xFFFFU);
            break;
        case 3U:
            register_set(&(TIM->CCR3), comp_value, 0xFFFFU);
            break;
        case 4U:
            register_set(&(TIM->CCR4), comp_value, 0xFFFFU);
            break;
        default:
            break;
    }
}