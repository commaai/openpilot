// this is needed for 1 mbps support
#define CAN_QUANTA 8
#define CAN_SEQ1 6 // roundf(quanta * 0.875f) - 1;
#define CAN_SEQ2 1 // roundf(quanta * 0.125f);

#define CAN_PCLK 24000
// 333 = 33.3 kbps
// 5000 = 500 kbps
#define can_speed_to_prescaler(x) (CAN_PCLK / CAN_QUANTA * 10 / (x))

bool llcan_set_speed(CAN_TypeDef *CAN, uint32_t speed, bool loopback, bool silent) {
  // initialization mode
  CAN->MCR = CAN_MCR_TTCM | CAN_MCR_INRQ;
  while((CAN->MSR & CAN_MSR_INAK) != CAN_MSR_INAK);

  // set time quanta from defines
  CAN->BTR = (CAN_BTR_TS1_0 * (CAN_SEQ1-1)) |
            (CAN_BTR_TS2_0 * (CAN_SEQ2-1)) |
            (can_speed_to_prescaler(speed) - 1);

  // silent loopback mode for debugging
  if (loopback) {
    CAN->BTR |= CAN_BTR_SILM | CAN_BTR_LBKM;
  }
  if (silent) {
    CAN->BTR |= CAN_BTR_SILM;
  }

  // reset
  CAN->MCR = CAN_MCR_TTCM | CAN_MCR_ABOM;

  #define CAN_TIMEOUT 1000000
  int tmp = 0;
  bool ret = false;
  while(((CAN->MSR & CAN_MSR_INAK) == CAN_MSR_INAK) && (tmp < CAN_TIMEOUT)) tmp++;
  if (tmp < CAN_TIMEOUT) {
    ret = true;
  }

  return ret;
}

void llcan_init(CAN_TypeDef *CAN) {
  // accept all filter
  CAN->FMR |= CAN_FMR_FINIT;

  // no mask
  CAN->sFilterRegister[0].FR1 = 0;
  CAN->sFilterRegister[0].FR2 = 0;
  CAN->sFilterRegister[14].FR1 = 0;
  CAN->sFilterRegister[14].FR2 = 0;
  CAN->FA1R |= 1 | (1U << 14);

  CAN->FMR &= ~(CAN_FMR_FINIT);

  // enable certain CAN interrupts
  CAN->IER |= CAN_IER_TMEIE | CAN_IER_FMPIE0 |  CAN_IER_WKUIE;

  if (CAN == CAN1) {
    NVIC_EnableIRQ(CAN1_TX_IRQn);
    NVIC_EnableIRQ(CAN1_RX0_IRQn);
    NVIC_EnableIRQ(CAN1_SCE_IRQn);
  } else if (CAN == CAN2) {
    NVIC_EnableIRQ(CAN2_TX_IRQn);
    NVIC_EnableIRQ(CAN2_RX0_IRQn);
    NVIC_EnableIRQ(CAN2_SCE_IRQn);
#ifdef CAN3
  } else if (CAN == CAN3) {
    NVIC_EnableIRQ(CAN3_TX_IRQn);
    NVIC_EnableIRQ(CAN3_RX0_IRQn);
    NVIC_EnableIRQ(CAN3_SCE_IRQn);
#endif
  }
}

void llcan_clear_send(CAN_TypeDef *CAN) {
  CAN->TSR |= CAN_TSR_ABRQ0;
  CAN->MSR &= ~(CAN_MSR_ERRI);
  CAN->MSR = CAN->MSR;
}

