// SAE J2284-4 document specifies a bus-line network running at 2 Mbit/s
// SAE J2284-5 document specifies a point-to-point communication running at 5 Mbit/s

#define CAN_PCLK 80000U // KHz, sourced from PLL1Q
#define BITRATE_PRESCALER 2U // Valid from 250Kbps to 5Mbps with 80Mhz clock
#define CAN_SP_NOMINAL 80U // 80% for both SAE J2284-4 and SAE J2284-5
#define CAN_SP_DATA_2M 80U // 80% for SAE J2284-4
#define CAN_SP_DATA_5M 75U // 75% for SAE J2284-5
#define CAN_QUANTA(speed, prescaler) (CAN_PCLK / ((speed) / 10U * (prescaler)))
#define CAN_SEG1(tq, sp) (((tq) * (sp) / 100U)- 1U)
#define CAN_SEG2(tq, sp) ((tq) * (100U - (sp)) / 100U)

// FDCAN core settings
#define FDCAN_MESSAGE_RAM_SIZE 0x2800UL
#define FDCAN_START_ADDRESS 0x4000AC00UL
#define FDCAN_OFFSET 3384UL // bytes for each FDCAN module, equally
#define FDCAN_OFFSET_W 846UL // words for each FDCAN module, equally
#define FDCAN_END_ADDRESS 0x4000D3FCUL // Message RAM has a width of 4 bytes

// FDCAN_RX_FIFO_0_EL_CNT + FDCAN_TX_FIFO_EL_CNT can't exceed 47 elements (47 * 72 bytes = 3,384 bytes) per FDCAN module

// RX FIFO 0
#define FDCAN_RX_FIFO_0_EL_CNT 46UL
#define FDCAN_RX_FIFO_0_HEAD_SIZE 8UL // bytes
#define FDCAN_RX_FIFO_0_DATA_SIZE 64UL // bytes
#define FDCAN_RX_FIFO_0_EL_SIZE (FDCAN_RX_FIFO_0_HEAD_SIZE + FDCAN_RX_FIFO_0_DATA_SIZE)
#define FDCAN_RX_FIFO_0_EL_W_SIZE (FDCAN_RX_FIFO_0_EL_SIZE / 4UL)
#define FDCAN_RX_FIFO_0_OFFSET 0UL

// TX FIFO
#define FDCAN_TX_FIFO_EL_CNT 1UL
#define FDCAN_TX_FIFO_HEAD_SIZE 8UL // bytes
#define FDCAN_TX_FIFO_DATA_SIZE 64UL // bytes
#define FDCAN_TX_FIFO_EL_SIZE (FDCAN_TX_FIFO_HEAD_SIZE + FDCAN_TX_FIFO_DATA_SIZE)
#define FDCAN_TX_FIFO_EL_W_SIZE (FDCAN_TX_FIFO_EL_SIZE / 4UL)
#define FDCAN_TX_FIFO_OFFSET (FDCAN_RX_FIFO_0_OFFSET + (FDCAN_RX_FIFO_0_EL_CNT * FDCAN_RX_FIFO_0_EL_W_SIZE))

#define CAN_NAME_FROM_CANIF(CAN_DEV) (((CAN_DEV)==FDCAN1) ? "FDCAN1" : (((CAN_DEV) == FDCAN2) ? "FDCAN2" : "FDCAN3"))
#define CAN_NUM_FROM_CANIF(CAN_DEV) (((CAN_DEV)==FDCAN1) ? 0UL : (((CAN_DEV) == FDCAN2) ? 1UL : 2UL))


void print(const char *a);

// kbps multiplied by 10
const uint32_t speeds[] = {100U, 200U, 500U, 1000U, 1250U, 2500U, 5000U, 10000U};
const uint32_t data_speeds[] = {100U, 200U, 500U, 1000U, 1250U, 2500U, 5000U, 10000U, 20000U, 50000U};


bool fdcan_request_init(FDCAN_GlobalTypeDef *FDCANx) {
  bool ret = true;
  // Exit from sleep mode
  FDCANx->CCCR &= ~(FDCAN_CCCR_CSR);
  while ((FDCANx->CCCR & FDCAN_CCCR_CSA) == FDCAN_CCCR_CSA);

  // Request init
  uint32_t timeout_counter = 0U;
  FDCANx->CCCR |= FDCAN_CCCR_INIT;
  while ((FDCANx->CCCR & FDCAN_CCCR_INIT) == 0U) {
    // Delay for about 1ms
    delay(10000);
    timeout_counter++;

    if (timeout_counter >= CAN_INIT_TIMEOUT_MS){
      ret = false;
      break;
    }
  }
  return ret;
}

bool fdcan_exit_init(FDCAN_GlobalTypeDef *FDCANx) {
  bool ret = true;

  FDCANx->CCCR &= ~(FDCAN_CCCR_INIT);
  uint32_t timeout_counter = 0U;
  while ((FDCANx->CCCR & FDCAN_CCCR_INIT) != 0U) {
    // Delay for about 1ms
    delay(10000);
    timeout_counter++;

    if (timeout_counter >= CAN_INIT_TIMEOUT_MS) {
      ret = false;
      break;
    }
  }
  return ret;
}

bool llcan_set_speed(FDCAN_GlobalTypeDef *FDCANx, uint32_t speed, uint32_t data_speed, bool non_iso, bool loopback, bool silent) {
  UNUSED(speed);
  bool ret = fdcan_request_init(FDCANx);

  if (ret) {
    // Enable config change
    FDCANx->CCCR |= FDCAN_CCCR_CCE;

    //Reset operation mode to Normal
    FDCANx->CCCR &= ~(FDCAN_CCCR_TEST);
    FDCANx->TEST &= ~(FDCAN_TEST_LBCK);
    FDCANx->CCCR &= ~(FDCAN_CCCR_MON);
    FDCANx->CCCR &= ~(FDCAN_CCCR_ASM);
    FDCANx->CCCR &= ~(FDCAN_CCCR_NISO);

    // TODO: add as a separate safety mode
    // Enable ASM restricted operation(for debug or automatic bitrate switching)
    //FDCANx->CCCR |= FDCAN_CCCR_ASM;

    uint8_t prescaler = BITRATE_PRESCALER;
    if (speed < 2500U) {
      // The only way to support speeds lower than 250Kbit/s (down to 10Kbit/s)
      prescaler = BITRATE_PRESCALER * 16U;
    }

    // Set the nominal bit timing values
    uint32_t tq = CAN_QUANTA(speed, prescaler);
    uint32_t sp = CAN_SP_NOMINAL;
    uint32_t seg1 = CAN_SEG1(tq, sp);
    uint32_t seg2 = CAN_SEG2(tq, sp);
    uint8_t sjw = MIN(127U, seg2);

    FDCANx->NBTP = (((sjw & 0x7FUL)-1U)<<FDCAN_NBTP_NSJW_Pos) | (((seg1 & 0xFFU)-1U)<<FDCAN_NBTP_NTSEG1_Pos) | (((seg2 & 0x7FU)-1U)<<FDCAN_NBTP_NTSEG2_Pos) | (((prescaler & 0x1FFUL)-1U)<<FDCAN_NBTP_NBRP_Pos);

    // Set the data bit timing values
    if (data_speed == 50000U) {
      sp = CAN_SP_DATA_5M;
    } else {
      sp = CAN_SP_DATA_2M;
    }
    tq = CAN_QUANTA(data_speed, prescaler);
    seg1 = CAN_SEG1(tq, sp);
    seg2 = CAN_SEG2(tq, sp);
    sjw = MIN(15U, seg2);

    FDCANx->DBTP = (((sjw & 0xFUL)-1U)<<FDCAN_DBTP_DSJW_Pos) | (((seg1 & 0x1FU)-1U)<<FDCAN_DBTP_DTSEG1_Pos) | (((seg2 & 0xFU)-1U)<<FDCAN_DBTP_DTSEG2_Pos) | (((prescaler & 0x1FUL)-1U)<<FDCAN_DBTP_DBRP_Pos);

    if (non_iso) {
      // FD non-ISO mode
      FDCANx->CCCR |= FDCAN_CCCR_NISO;
    }

    // Silent loopback is known as internal loopback in the docs
    if (loopback) {
      FDCANx->CCCR |= FDCAN_CCCR_TEST;
      FDCANx->TEST |= FDCAN_TEST_LBCK;
      FDCANx->CCCR |= FDCAN_CCCR_MON;
    }
    // Silent is known as bus monitoring in the docs
    if (silent) {
      FDCANx->CCCR |= FDCAN_CCCR_MON;
    }
    ret = fdcan_exit_init(FDCANx);
    if (!ret) {
      print(CAN_NAME_FROM_CANIF(FDCANx)); print(" set_speed timed out! (2)\n");
    }
  } else {
    print(CAN_NAME_FROM_CANIF(FDCANx)); print(" set_speed timed out! (1)\n");
  }
  return ret;
}

void llcan_irq_disable(const FDCAN_GlobalTypeDef *FDCANx) {
  if (FDCANx == FDCAN1) {
    NVIC_DisableIRQ(FDCAN1_IT0_IRQn);
    NVIC_DisableIRQ(FDCAN1_IT1_IRQn);
  } else if (FDCANx == FDCAN2) {
    NVIC_DisableIRQ(FDCAN2_IT0_IRQn);
    NVIC_DisableIRQ(FDCAN2_IT1_IRQn);
  } else if (FDCANx == FDCAN3) {
    NVIC_DisableIRQ(FDCAN3_IT0_IRQn);
    NVIC_DisableIRQ(FDCAN3_IT1_IRQn);
  } else {
  }
}

void llcan_irq_enable(const FDCAN_GlobalTypeDef *FDCANx) {
  if (FDCANx == FDCAN1) {
    NVIC_EnableIRQ(FDCAN1_IT0_IRQn);
    NVIC_EnableIRQ(FDCAN1_IT1_IRQn);
  } else if (FDCANx == FDCAN2) {
    NVIC_EnableIRQ(FDCAN2_IT0_IRQn);
    NVIC_EnableIRQ(FDCAN2_IT1_IRQn);
  } else if (FDCANx == FDCAN3) {
    NVIC_EnableIRQ(FDCAN3_IT0_IRQn);
    NVIC_EnableIRQ(FDCAN3_IT1_IRQn);
  } else {
  }
}

bool llcan_init(FDCAN_GlobalTypeDef *FDCANx) {
  uint32_t can_number = CAN_NUM_FROM_CANIF(FDCANx);
  bool ret = fdcan_request_init(FDCANx);

  if (ret) {
    // Enable config change
    FDCANx->CCCR |= FDCAN_CCCR_CCE;
    // Enable automatic retransmission
    FDCANx->CCCR &= ~(FDCAN_CCCR_DAR);
    // Enable transmission pause feature
    FDCANx->CCCR |= FDCAN_CCCR_TXP;
    // Disable protocol exception handling
    FDCANx->CCCR |= FDCAN_CCCR_PXHD;
    // FD with BRS
    FDCANx->CCCR |= (FDCAN_CCCR_FDOE | FDCAN_CCCR_BRSE);

    // Set TX mode to FIFO
    FDCANx->TXBC &= ~(FDCAN_TXBC_TFQM);
    // Configure TX element data size
    FDCANx->TXESC |= 0x7U << FDCAN_TXESC_TBDS_Pos; // 64 bytes
    //Configure RX FIFO0 element data size
    FDCANx->RXESC |= 0x7U << FDCAN_RXESC_F0DS_Pos;
    // Disable filtering, accept all valid frames received
    FDCANx->XIDFC &= ~(FDCAN_XIDFC_LSE); // No extended filters
    FDCANx->SIDFC &= ~(FDCAN_SIDFC_LSS); // No standard filters
    FDCANx->GFC &= ~(FDCAN_GFC_RRFE); // Accept extended remote frames
    FDCANx->GFC &= ~(FDCAN_GFC_RRFS); // Accept standard remote frames
    FDCANx->GFC &= ~(FDCAN_GFC_ANFE); // Accept extended frames to FIFO 0
    FDCANx->GFC &= ~(FDCAN_GFC_ANFS); // Accept standard frames to FIFO 0

    uint32_t RxFIFO0SA = FDCAN_START_ADDRESS + (can_number * FDCAN_OFFSET);
    uint32_t TxFIFOSA = RxFIFO0SA + (FDCAN_RX_FIFO_0_EL_CNT * FDCAN_RX_FIFO_0_EL_SIZE);

    // RX FIFO 0
    FDCANx->RXF0C |= (FDCAN_RX_FIFO_0_OFFSET + (can_number * FDCAN_OFFSET_W)) << FDCAN_RXF0C_F0SA_Pos;
    FDCANx->RXF0C |= FDCAN_RX_FIFO_0_EL_CNT << FDCAN_RXF0C_F0S_Pos;
    // RX FIFO 0 switch to non-blocking (overwrite) mode
    FDCANx->RXF0C |= FDCAN_RXF0C_F0OM;

    // TX FIFO (mode set earlier)
    FDCANx->TXBC |= (FDCAN_TX_FIFO_OFFSET + (can_number * FDCAN_OFFSET_W)) << FDCAN_TXBC_TBSA_Pos;
    FDCANx->TXBC |= FDCAN_TX_FIFO_EL_CNT << FDCAN_TXBC_TFQS_Pos;

    // Flush allocated RAM
    uint32_t EndAddress = TxFIFOSA + (FDCAN_TX_FIFO_EL_CNT * FDCAN_TX_FIFO_EL_SIZE);
    for (uint32_t RAMcounter = RxFIFO0SA; RAMcounter < EndAddress; RAMcounter += 4U) {
        *(uint32_t *)(RAMcounter) = 0x00000000;
    }

    // Enable both interrupts for each module
    FDCANx->ILE = (FDCAN_ILE_EINT0 | FDCAN_ILE_EINT1);

    FDCANx->IE &= 0x0U; // Reset all interrupts
    // Messages for INT0
    FDCANx->IE |= FDCAN_IE_RF0NE; // Rx FIFO 0 new message
    FDCANx->IE |= FDCAN_IE_PEDE | FDCAN_IE_PEAE | FDCAN_IE_BOE | FDCAN_IE_EPE | FDCAN_IE_RF0LE;

    // Messages for INT1 (Only TFE works??)
    FDCANx->ILS |= FDCAN_ILS_TFEL;
    FDCANx->IE |= FDCAN_IE_TFEE; // Tx FIFO empty

    ret = fdcan_exit_init(FDCANx);
    if(!ret) {
      print(CAN_NAME_FROM_CANIF(FDCANx)); print(" llcan_init timed out (2)!\n");
    }

    llcan_irq_enable(FDCANx);

  } else {
    print(CAN_NAME_FROM_CANIF(FDCANx)); print(" llcan_init timed out (1)!\n");
  }
  return ret;
}

void llcan_clear_send(FDCAN_GlobalTypeDef *FDCANx) {
  // from datasheet: "Transmit cancellation is not intended for Tx FIFO operation."
  // so we need to clear pending transmission manually by resetting FDCAN core
  FDCANx->IR |= 0x3FCFFFFFU; // clear all interrupts
  bool ret = llcan_init(FDCANx);
  UNUSED(ret);
}
