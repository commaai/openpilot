// SAE 2284-3 : minimum 16 tq, SJW 3, sample point at 81.3%
#define CAN_QUANTA 16U
#define CAN_SEQ1 12U
#define CAN_SEQ2 3U
#define CAN_SJW  3U

#define CAN_PCLK (CORE_FREQ / 2U / 1000U)
#define can_speed_to_prescaler(x) (CAN_PCLK / CAN_QUANTA * 10U / (x))
#define CAN_INIT_TIMEOUT_MS 500

bool llcan_set_speed(CAN_TypeDef *CAN_obj, uint32_t speed, bool loopback, bool silent) {
  bool ret = true;

  // initialization mode
  CAN1->MCR = CAN_MCR_INRQ; // When we want to use only CAN2 - need to do that
  CAN_obj->MCR = CAN_MCR_INRQ;
  uint32_t timeout_counter = 0U;
  while((CAN_obj->MSR & CAN_MSR_INAK) != CAN_MSR_INAK){
    // Delay for about 1ms
    delay(10000);
    timeout_counter++;

    if(timeout_counter >= CAN_INIT_TIMEOUT_MS){
      ret = false;
      break;
    }
  }

  if(ret){
    // set time quanta from defines
    CAN_obj->BTR = ((CAN_BTR_TS1_0 * (CAN_SEQ1-1)) |
                    (CAN_BTR_TS2_0 * (CAN_SEQ2-1)) |
                    (CAN_BTR_SJW_0 * (CAN_SJW-1)) |
                    (can_speed_to_prescaler(speed) - 1U));

    // silent loopback mode for debugging
    if (loopback) {
      CAN_obj->BTR |= CAN_BTR_SILM | CAN_BTR_LBKM;
    }
    if (silent) {
      CAN_obj->BTR |= CAN_BTR_SILM;
    }

    CAN_obj->MCR |= CAN_MCR_AWUM; // Automatic wakeup mode
    CAN_obj->MCR |= CAN_MCR_ABOM; // Automatic bus-off management
    CAN_obj->MCR &= ~CAN_MCR_NART; // Automatic retransmission
    CAN_obj->MCR |= CAN_MCR_TXFP; // Priority driven by the request order (chronologically)
    CAN_obj->MCR &= ~CAN_MCR_INRQ;

    timeout_counter = 0U;
    while(((CAN_obj->MSR & CAN_MSR_INAK) == CAN_MSR_INAK)) {
      // Delay for about 1ms
      delay(10000);
      timeout_counter++;

      if(timeout_counter >= CAN_INIT_TIMEOUT_MS){
        ret = false;
        break;
      }
    }
  }

  return ret;
}

bool llcan_init(CAN_TypeDef *CAN_obj) {
  bool ret = true;

  // Enter init mode
  CAN_obj->FMR |= CAN_FMR_FINIT;

  // Wait for INAK bit to be set
  uint32_t timeout_counter = 0U;
  while(((CAN_obj->MSR & CAN_MSR_INAK) == CAN_MSR_INAK)) {
    // Delay for about 1ms
    delay(10000);
    timeout_counter++;

    if(timeout_counter >= CAN_INIT_TIMEOUT_MS){
      ret = false;
      break;
    }
  }

  if(ret){
    // no mask
    // For some weird reason some of these registers do not want to set properly on CAN2 and CAN3. Probably something to do with the single/dual mode and their different filters.
    // Filters MUST be set always through CAN1(Master) as CAN2/3 are Slave
    CAN1->sFilterRegister[0].FR1 = 0U;
    CAN1->sFilterRegister[0].FR2 = 0U;
    CAN1->sFilterRegister[14].FR1 = 0U;
    CAN1->sFilterRegister[14].FR2 = 0U;
    CAN1->FA1R |= 1U | (1U << 14);

    // Exit init mode, do not wait
    CAN_obj->FMR &= ~CAN_FMR_FINIT;

    // enable certain CAN interrupts
    CAN1->IER = 0U; // When we want to use only CAN2 - need to do that
    CAN_obj->IER = CAN_IER_FMPIE0 | CAN_IER_TMEIE | CAN_IER_WKUIE;

    if (CAN_obj == CAN1) {
      HAL_NVIC_EnableIRQ(CAN1_TX_IRQn);
      HAL_NVIC_SetPriority(CAN1_RX0_IRQn, 0, 0);
      HAL_NVIC_EnableIRQ(CAN1_RX0_IRQn);
      HAL_NVIC_SetPriority(CAN1_SCE_IRQn, 0, 0);
      HAL_NVIC_EnableIRQ(CAN1_SCE_IRQn);
    } else {
      HAL_NVIC_SetPriority(CAN2_TX_IRQn, 0, 0);
      HAL_NVIC_EnableIRQ(CAN2_TX_IRQn);
      HAL_NVIC_SetPriority(CAN2_RX0_IRQn, 0, 0);
      HAL_NVIC_EnableIRQ(CAN2_RX0_IRQn);
      HAL_NVIC_SetPriority(CAN2_SCE_IRQn, 0, 0);
      HAL_NVIC_EnableIRQ(CAN2_SCE_IRQn);
    }
  }
  return ret;
}

void llcan_clear_send(CAN_TypeDef *CAN_obj) {
  CAN_obj->TSR |= CAN_TSR_ABRQ0;
  CAN_obj->MSR &= ~CAN_MSR_ERRI;
  // cppcheck-suppress selfAssignment ; needed to clear the register
  CAN_obj->MSR = CAN_obj->MSR;
}
