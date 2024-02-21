// Flasher and pedal use raw mailbox access
#define GET_MAILBOX_BYTE(msg, b) (((int)(b) > 3) ? (((msg)->RDHR >> (8U * ((unsigned int)(b) % 4U))) & 0xFFU) : (((msg)->RDLR >> (8U * (unsigned int)(b))) & 0xFFU))
#define GET_MAILBOX_BYTES_04(msg) ((msg)->RDLR)
#define GET_MAILBOX_BYTES_48(msg) ((msg)->RDHR)

// SAE 2284-3 : minimum 16 tq, SJW 3, sample point at 81.3%
#define CAN_QUANTA 16U
#define CAN_SEQ1 12U
#define CAN_SEQ2 3U
#define CAN_SJW  3U

#define CAN_PCLK 48000U
// 333 = 33.3 kbps
// 5000 = 500 kbps
#define can_speed_to_prescaler(x) (CAN_PCLK / CAN_QUANTA * 10U / (x))

#define CAN_NAME_FROM_CANIF(CAN_DEV) (((CAN_DEV)==CAN1) ? "CAN1" : (((CAN_DEV) == CAN2) ? "CAN2" : "CAN3"))

void print(const char *a);

// kbps multiplied by 10
const uint32_t speeds[] = {100U, 200U, 500U, 1000U, 1250U, 2500U, 5000U, 10000U};
const uint32_t data_speeds[] = {0U}; // No separate data speed, dummy

bool llcan_set_speed(CAN_TypeDef *CANx, uint32_t speed, bool loopback, bool silent) {
  bool ret = true;

  // initialization mode
  register_set(&(CANx->MCR), CAN_MCR_TTCM | CAN_MCR_INRQ, 0x180FFU);
  uint32_t timeout_counter = 0U;
  while((CANx->MSR & CAN_MSR_INAK) != CAN_MSR_INAK){
    // Delay for about 1ms
    delay(10000);
    timeout_counter++;

    if(timeout_counter >= CAN_INIT_TIMEOUT_MS){
      print(CAN_NAME_FROM_CANIF(CANx)); print(" set_speed timed out (1)!\n");
      ret = false;
      break;
    }
  }

  if(ret){
    // set time quanta from defines
    register_set(&(CANx->BTR), ((CAN_BTR_TS1_0 * (CAN_SEQ1-1U)) |
                                   (CAN_BTR_TS2_0 * (CAN_SEQ2-1U)) |
                                   (CAN_BTR_SJW_0 * (CAN_SJW-1U)) |
                                   (can_speed_to_prescaler(speed) - 1U)), 0xC37F03FFU);

    // silent loopback mode for debugging
    if (loopback) {
      register_set_bits(&(CANx->BTR), CAN_BTR_SILM | CAN_BTR_LBKM);
    }
    if (silent) {
      register_set_bits(&(CANx->BTR), CAN_BTR_SILM);
    }

    // reset
    register_set(&(CANx->MCR), CAN_MCR_TTCM | CAN_MCR_ABOM, 0x180FFU);

    timeout_counter = 0U;
    while(((CANx->MSR & CAN_MSR_INAK) == CAN_MSR_INAK)) {
      // Delay for about 1ms
      delay(10000);
      timeout_counter++;

      if(timeout_counter >= CAN_INIT_TIMEOUT_MS){
        print(CAN_NAME_FROM_CANIF(CANx)); print(" set_speed timed out (2)!\n");
        ret = false;
        break;
      }
    }
  }

  return ret;
}

void llcan_irq_disable(const CAN_TypeDef *CANx) {
  if (CANx == CAN1) {
    NVIC_DisableIRQ(CAN1_TX_IRQn);
    NVIC_DisableIRQ(CAN1_RX0_IRQn);
    NVIC_DisableIRQ(CAN1_SCE_IRQn);
  } else if (CANx == CAN2) {
    NVIC_DisableIRQ(CAN2_TX_IRQn);
    NVIC_DisableIRQ(CAN2_RX0_IRQn);
    NVIC_DisableIRQ(CAN2_SCE_IRQn);
  } else if (CANx == CAN3) {
    NVIC_DisableIRQ(CAN3_TX_IRQn);
    NVIC_DisableIRQ(CAN3_RX0_IRQn);
    NVIC_DisableIRQ(CAN3_SCE_IRQn);
  } else {
  }
}

void llcan_irq_enable(const CAN_TypeDef *CANx) {
  if (CANx == CAN1) {
    NVIC_EnableIRQ(CAN1_TX_IRQn);
    NVIC_EnableIRQ(CAN1_RX0_IRQn);
    NVIC_EnableIRQ(CAN1_SCE_IRQn);
  } else if (CANx == CAN2) {
    NVIC_EnableIRQ(CAN2_TX_IRQn);
    NVIC_EnableIRQ(CAN2_RX0_IRQn);
    NVIC_EnableIRQ(CAN2_SCE_IRQn);
  } else if (CANx == CAN3) {
    NVIC_EnableIRQ(CAN3_TX_IRQn);
    NVIC_EnableIRQ(CAN3_RX0_IRQn);
    NVIC_EnableIRQ(CAN3_SCE_IRQn);
  } else {
  }
}

bool llcan_init(CAN_TypeDef *CANx) {
  bool ret = true;

  // Enter init mode
  register_set_bits(&(CANx->FMR), CAN_FMR_FINIT);

  // Wait for INAK bit to be set
  uint32_t timeout_counter = 0U;
  while(((CANx->MSR & CAN_MSR_INAK) == CAN_MSR_INAK)) {
    // Delay for about 1ms
    delay(10000);
    timeout_counter++;

    if(timeout_counter >= CAN_INIT_TIMEOUT_MS){
      print(CAN_NAME_FROM_CANIF(CANx)); print(" initialization timed out!\n");
      ret = false;
      break;
    }
  }

  if(ret){
    // no mask
    // For some weird reason some of these registers do not want to set properly on CAN2 and CAN3. Probably something to do with the single/dual mode and their different filters.
    CANx->sFilterRegister[0].FR1 = 0U;
    CANx->sFilterRegister[0].FR2 = 0U;
    CANx->sFilterRegister[14].FR1 = 0U;
    CANx->sFilterRegister[14].FR2 = 0U;
    CANx->FA1R |= 1U | (1UL << 14);

    // Exit init mode, do not wait
    register_clear_bits(&(CANx->FMR), CAN_FMR_FINIT);

    // enable certain CAN interrupts
    register_set_bits(&(CANx->IER), CAN_IER_TMEIE | CAN_IER_FMPIE0 | CAN_IER_ERRIE | CAN_IER_LECIE | CAN_IER_BOFIE | CAN_IER_EPVIE | CAN_IER_EWGIE | CAN_IER_FOVIE0 | CAN_IER_FFIE0);

    // clear overrun flag on init
    CANx->RF0R &= ~(CAN_RF0R_FOVR0);

    llcan_irq_enable(CANx);
  }
  return ret;
}

void llcan_clear_send(CAN_TypeDef *CANx) {
  CANx->TSR |= CAN_TSR_ABRQ0; // Abort message transmission on error interrupt
  CANx->MSR |= CAN_MSR_ERRI; // Clear error interrupt
}
