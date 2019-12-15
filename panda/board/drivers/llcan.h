// this is needed for 1 mbps support
#define CAN_QUANTA 8U
#define CAN_SEQ1 6 // roundf(quanta * 0.875f) - 1;
#define CAN_SEQ2 1 // roundf(quanta * 0.125f);

#define CAN_PCLK 24000U
// 333 = 33.3 kbps
// 5000 = 500 kbps
#define can_speed_to_prescaler(x) (CAN_PCLK / CAN_QUANTA * 10U / (x))

#define GET_BUS(msg) (((msg)->RDTR >> 4) & 0xFF)
#define GET_LEN(msg) ((msg)->RDTR & 0xF)
#define GET_ADDR(msg) ((((msg)->RIR & 4) != 0) ? ((msg)->RIR >> 3) : ((msg)->RIR >> 21))
#define GET_BYTE(msg, b) (((int)(b) > 3) ? (((msg)->RDHR >> (8U * ((unsigned int)(b) % 4U))) & 0XFFU) : (((msg)->RDLR >> (8U * (unsigned int)(b))) & 0xFFU))
#define GET_BYTES_04(msg) ((msg)->RDLR)
#define GET_BYTES_48(msg) ((msg)->RDHR)

void puts(const char *a);

bool llcan_set_speed(CAN_TypeDef *CAN_obj, uint32_t speed, bool loopback, bool silent) {
  // initialization mode
  CAN_obj->MCR = CAN_MCR_TTCM | CAN_MCR_INRQ;
  while((CAN_obj->MSR & CAN_MSR_INAK) != CAN_MSR_INAK);

  // set time quanta from defines
  CAN_obj->BTR = (CAN_BTR_TS1_0 * (CAN_SEQ1-1)) |
            (CAN_BTR_TS2_0 * (CAN_SEQ2-1)) |
            (can_speed_to_prescaler(speed) - 1U);

  // silent loopback mode for debugging
  if (loopback) {
    CAN_obj->BTR |= CAN_BTR_SILM | CAN_BTR_LBKM;
  }
  if (silent) {
    CAN_obj->BTR |= CAN_BTR_SILM;
  }

  // reset
  // cppcheck-suppress redundantAssignment ; it's a register
  CAN_obj->MCR = CAN_MCR_TTCM | CAN_MCR_ABOM;

  #define CAN_TIMEOUT 1000000
  int tmp = 0;
  bool ret = false;
  while(((CAN_obj->MSR & CAN_MSR_INAK) == CAN_MSR_INAK) && (tmp < CAN_TIMEOUT)) tmp++;
  if (tmp < CAN_TIMEOUT) {
    ret = true;
  }

  return ret;
}

void llcan_init(CAN_TypeDef *CAN_obj) {
  // accept all filter
  CAN_obj->FMR |= CAN_FMR_FINIT;

  // no mask
  CAN_obj->sFilterRegister[0].FR1 = 0;
  CAN_obj->sFilterRegister[0].FR2 = 0;
  CAN_obj->sFilterRegister[14].FR1 = 0;
  CAN_obj->sFilterRegister[14].FR2 = 0;
  CAN_obj->FA1R |= 1U | (1U << 14);

  CAN_obj->FMR &= ~(CAN_FMR_FINIT);

  // enable certain CAN interrupts
  CAN_obj->IER |= CAN_IER_TMEIE | CAN_IER_FMPIE0 |  CAN_IER_WKUIE;

  if (CAN_obj == CAN1) {
    NVIC_EnableIRQ(CAN1_TX_IRQn);
    NVIC_EnableIRQ(CAN1_RX0_IRQn);
    NVIC_EnableIRQ(CAN1_SCE_IRQn);
  } else if (CAN_obj == CAN2) {
    NVIC_EnableIRQ(CAN2_TX_IRQn);
    NVIC_EnableIRQ(CAN2_RX0_IRQn);
    NVIC_EnableIRQ(CAN2_SCE_IRQn);
#ifdef CAN3
  } else if (CAN_obj == CAN3) {
    NVIC_EnableIRQ(CAN3_TX_IRQn);
    NVIC_EnableIRQ(CAN3_RX0_IRQn);
    NVIC_EnableIRQ(CAN3_SCE_IRQn);
#endif
  } else {
    puts("Invalid CAN: initialization failed\n");
  }
}

void llcan_clear_send(CAN_TypeDef *CAN_obj) {
  CAN_obj->TSR |= CAN_TSR_ABRQ0;
  CAN_obj->MSR &= ~(CAN_MSR_ERRI);
  // cppcheck-suppress selfAssignment ; needed to clear the register
  CAN_obj->MSR = CAN_obj->MSR;
}

