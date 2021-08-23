#define FDCAN_MESSAGE_RAM_SIZE 0x2800UL
#define FDCAN_START_ADDRESS 0x4000AC00UL
#define FDCAN_OFFSET 3412UL // bytes for each FDCAN module
#define FDCAN_OFFSET_W 853UL // words for each FDCAN module
#define FDCAN_END_ADDRESS 0x4000D3FCUL // Message RAM has a width of 4 Bytes

// With this settings we can go up to 6Mbit/s
#define CAN_SYNC_JW     1U // 1 to 4
#define CAN_PHASE_SEG1  6U // =(PROP_SEG + PHASE_SEG1) , 1 to 16
#define CAN_PHASE_SEG2  1U // 1 to 8
#define CAN_PCLK 48000U // Sourced from PLL1Q
#define CAN_QUANTA (1U + CAN_PHASE_SEG1 + CAN_PHASE_SEG2)
// Valid speeds in kbps and their prescalers:
// 10=600, 20=300, 50=120, 83.333=72, 100=60, 125=48, 250=24, 500=12, 1000=6, 2000=3, 3000=2, 6000=1
#define can_speed_to_prescaler(x) (CAN_PCLK / CAN_QUANTA * 10U / (x))

// RX FIFO 0
#define FDCAN_RX_FIFO_0_EL_CNT 32UL
#define FDCAN_RX_FIFO_0_HEAD_SIZE 8UL // bytes
#define FDCAN_RX_FIFO_0_DATA_SIZE 8UL // bytes
#define FDCAN_RX_FIFO_0_EL_SIZE (FDCAN_RX_FIFO_0_HEAD_SIZE + FDCAN_RX_FIFO_0_DATA_SIZE)
#define FDCAN_RX_FIFO_0_EL_W_SIZE (FDCAN_RX_FIFO_0_EL_SIZE / 4UL)
#define FDCAN_RX_FIFO_0_OFFSET 0UL

// TX FIFO
#define FDCAN_TX_FIFO_EL_CNT 32UL
#define FDCAN_TX_FIFO_HEAD_SIZE 8UL // bytes
#define FDCAN_TX_FIFO_DATA_SIZE 8UL // bytes
#define FDCAN_TX_FIFO_EL_SIZE (FDCAN_TX_FIFO_HEAD_SIZE + FDCAN_TX_FIFO_DATA_SIZE)
#define FDCAN_TX_FIFO_EL_W_SIZE (FDCAN_TX_FIFO_EL_SIZE / 4UL)
#define FDCAN_TX_FIFO_OFFSET (FDCAN_RX_FIFO_0_OFFSET + (FDCAN_RX_FIFO_0_EL_CNT * FDCAN_RX_FIFO_0_EL_W_SIZE))

#define CAN_NAME_FROM_CANIF(CAN_DEV) (((CAN_DEV)==FDCAN1) ? "FDCAN1" : (((CAN_DEV) == FDCAN2) ? "FDCAN2" : "FDCAN3"))
#define CAN_NUM_FROM_CANIF(CAN_DEV) (((CAN_DEV)==FDCAN1) ? 0UL : (((CAN_DEV) == FDCAN2) ? 1UL : 2UL))

// For backwards compatibility with safety code
typedef struct {
  __IO uint32_t RIR;  /*!< CAN receive FIFO mailbox identifier register */
  __IO uint32_t RDTR; /*!< CAN receive FIFO mailbox data length control and time stamp register */
  __IO uint32_t RDLR; /*!< CAN receive FIFO mailbox data low register */
  __IO uint32_t RDHR; /*!< CAN receive FIFO mailbox data high register */
} CAN_FIFOMailBox_TypeDef;

void puts(const char *a);

bool fdcan_request_init(FDCAN_GlobalTypeDef *CANx) {
  bool ret = true;
  // Exit from sleep mode
  CANx->CCCR &= ~(FDCAN_CCCR_CSR);
  while ((CANx->CCCR & FDCAN_CCCR_CSA) == FDCAN_CCCR_CSA);

  // Request init
  uint32_t timeout_counter = 0U;
  CANx->CCCR |= FDCAN_CCCR_INIT;
  while ((CANx->CCCR & FDCAN_CCCR_INIT) == 0) {
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

bool fdcan_exit_init(FDCAN_GlobalTypeDef *CANx) {
  bool ret = true;

  CANx->CCCR &= ~(FDCAN_CCCR_INIT);
  uint32_t timeout_counter = 0U;
  while ((CANx->CCCR & FDCAN_CCCR_INIT) != 0) {
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

bool llcan_set_speed(FDCAN_GlobalTypeDef *CANx, uint32_t speed, uint32_t data_speed, bool loopback, bool silent) {
  bool ret = fdcan_request_init(CANx);

  if (ret) {
    // Enable config change
    CANx->CCCR |= FDCAN_CCCR_CCE;

    //Reset operation mode to Normal
    CANx->CCCR &= ~(FDCAN_CCCR_TEST);
    CANx->TEST &= ~(FDCAN_TEST_LBCK);
    CANx->CCCR &= ~(FDCAN_CCCR_MON);
    CANx->CCCR &= ~(FDCAN_CCCR_ASM);

    // Set the nominal bit timing register
    CANx->NBTP = ((CAN_SYNC_JW-1U)<<FDCAN_NBTP_NSJW_Pos) | ((CAN_PHASE_SEG1-1U)<<FDCAN_NBTP_NTSEG1_Pos) | ((CAN_PHASE_SEG2-1U)<<FDCAN_NBTP_NTSEG2_Pos) | ((can_speed_to_prescaler(speed)-1U)<<FDCAN_NBTP_NBRP_Pos);
    // Set the data bit timing register
    CANx->DBTP = ((CAN_SYNC_JW-1U)<<FDCAN_DBTP_DSJW_Pos) | ((CAN_PHASE_SEG1-1U)<<FDCAN_DBTP_DTSEG1_Pos) | ((CAN_PHASE_SEG2-1U)<<FDCAN_DBTP_DTSEG2_Pos) | ((can_speed_to_prescaler(data_speed)-1U)<<FDCAN_DBTP_DBRP_Pos);
    // Silent loopback is known as internal loopback in the docs
    if (loopback) {
      CANx->CCCR |= FDCAN_CCCR_TEST;
      CANx->TEST |= FDCAN_TEST_LBCK;
      CANx->CCCR |= FDCAN_CCCR_MON;
    }
    // Silent is known as bus monitoring in the docs
    if (silent) {
      CANx->CCCR |= FDCAN_CCCR_MON;
    }
    ret = fdcan_exit_init(CANx);
    if (!ret) {
      puts(CAN_NAME_FROM_CANIF(CANx)); puts(" set_speed timed out! (2)\n");
    }
  } else {
    puts(CAN_NAME_FROM_CANIF(CANx)); puts(" set_speed timed out! (1)\n");
  }
  return ret;
}

bool llcan_init(FDCAN_GlobalTypeDef *CANx) {
  uint32_t can_number = CAN_NUM_FROM_CANIF(CANx);
  bool ret = fdcan_request_init(CANx);

  if (ret) {
    // Enable config change
    CANx->CCCR |= FDCAN_CCCR_CCE;
    // Enable automatic retransmission
    CANx->CCCR &= ~(FDCAN_CCCR_DAR);
    // Enable transmission pause feature
    CANx->CCCR |= FDCAN_CCCR_TXP;
    // Disable protocol exception handling
    CANx->CCCR |= FDCAN_CCCR_PXHD;
    // FD with BRS
    CANx->CCCR |= (FDCAN_CCCR_FDOE | FDCAN_CCCR_BRSE);

    // Set TX mode to FIFO
    CANx->TXBC &= ~(FDCAN_TXBC_TFQM);
    // Configure TX element size (for now 8 bytes, no need to change)
    //CANx->TXESC |= 0x000U;
    //Configure RX FIFO0, FIFO1, RX buffer element sizes (no need for now, using classic 8 bytes)
    register_set(&(CANx->RXESC), 0x0U, (FDCAN_RXESC_F0DS | FDCAN_RXESC_F1DS | FDCAN_RXESC_RBDS));
    // Disable filtering, accept all valid frames received
    CANx->XIDFC &= ~(FDCAN_XIDFC_LSE); // No extended filters
    CANx->SIDFC &= ~(FDCAN_SIDFC_LSS); // No standard filters
    CANx->GFC &= ~(FDCAN_GFC_RRFE); // Accept extended remote frames
    CANx->GFC &= ~(FDCAN_GFC_RRFS); // Accept standard remote frames
    CANx->GFC &= ~(FDCAN_GFC_ANFE); // Accept extended frames to FIFO 0
    CANx->GFC &= ~(FDCAN_GFC_ANFS); // Accept standard frames to FIFO 0

    uint32_t RxFIFO0SA = FDCAN_START_ADDRESS + (can_number * FDCAN_OFFSET);
    uint32_t TxFIFOSA = RxFIFO0SA + (FDCAN_RX_FIFO_0_EL_CNT * FDCAN_RX_FIFO_0_EL_SIZE);

    // RX FIFO 0
    CANx->RXF0C = (FDCAN_RX_FIFO_0_OFFSET + (can_number * FDCAN_OFFSET_W)) << FDCAN_RXF0C_F0SA_Pos;
    CANx->RXF0C |= FDCAN_RX_FIFO_0_EL_CNT << FDCAN_RXF0C_F0S_Pos;
    // RX FIFO 0 switch to non-blocking (overwrite) mode
    CANx->RXF0C |= FDCAN_RXF0C_F0OM;

    // TX FIFO (mode set earlier)
    CANx->TXBC = (FDCAN_TX_FIFO_OFFSET + (can_number * FDCAN_OFFSET_W)) << FDCAN_TXBC_TBSA_Pos;
    CANx->TXBC |= FDCAN_TX_FIFO_EL_CNT << FDCAN_TXBC_TFQS_Pos;

    // Flush allocated RAM
    uint32_t EndAddress = TxFIFOSA + (FDCAN_TX_FIFO_EL_CNT * FDCAN_TX_FIFO_EL_SIZE);
    for (uint32_t RAMcounter = RxFIFO0SA; RAMcounter < EndAddress; RAMcounter += 4U) {
        *(uint32_t *)(RAMcounter) = 0x00000000;
    }

    // Enable both interrupts for each module
    CANx->ILE = (FDCAN_ILE_EINT0 | FDCAN_ILE_EINT1);

    CANx->IE &= 0x0U; // Reset all interrupts
    // Messages for INT0
    CANx->IE |= FDCAN_IE_RF0NE; // Rx FIFO 0 new message

    // Messages for INT1 (Only TFE works??)
    CANx->ILS |= FDCAN_ILS_TFEL;
    CANx->IE |= FDCAN_IE_TFEE; // Tx FIFO empty
    
    ret = fdcan_exit_init(CANx);
    if(!ret) {
      puts(CAN_NAME_FROM_CANIF(CANx)); puts(" llcan_init timed out (2)!\n");
    }

    if (CANx == FDCAN1) {
      NVIC_EnableIRQ(FDCAN1_IT0_IRQn);
      NVIC_EnableIRQ(FDCAN1_IT1_IRQn);
    } else if (CANx == FDCAN2) {
      NVIC_EnableIRQ(FDCAN2_IT0_IRQn);
      NVIC_EnableIRQ(FDCAN2_IT1_IRQn);
    } else if (CANx == FDCAN3) {
      NVIC_EnableIRQ(FDCAN3_IT0_IRQn);
      NVIC_EnableIRQ(FDCAN3_IT1_IRQn);
    } else {
      puts("Invalid CAN: initialization failed\n");
    }

  } else {
    puts(CAN_NAME_FROM_CANIF(CANx)); puts(" llcan_init timed out (1)!\n");
  }
  return ret;
}

void llcan_clear_send(FDCAN_GlobalTypeDef *CANx) {
  // From H7 datasheet: Transmit cancellation is not intended for Tx FIFO operation.
  UNUSED(CANx);
}
