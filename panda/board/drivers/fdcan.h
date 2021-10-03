// IRQs: FDCAN1_IT0, FDCAN1_IT1
//       FDCAN2_IT0, FDCAN2_IT1
//       FDCAN3_IT0, FDCAN3_IT1

#define BUS_OFF_FAIL_LIMIT 2U
uint8_t bus_off_err[] = {0U, 0U, 0U};

FDCAN_GlobalTypeDef *cans[] = {FDCAN1, FDCAN2, FDCAN3};

bool can_set_speed(uint8_t can_number) {
  bool ret = true;
  FDCAN_GlobalTypeDef *CANx = CANIF_FROM_CAN_NUM(can_number);
  uint8_t bus_number = BUS_NUM_FROM_CAN_NUM(can_number);

  ret &= llcan_set_speed(CANx, can_speed[bus_number], can_data_speed[bus_number], can_loopback, (unsigned int)(can_silent) & (1U << can_number));
  return ret;
}

void can_set_gmlan(uint8_t bus) {
  UNUSED(bus);
  puts("GMLAN not available on red panda\n");
}

void cycle_transceiver(uint8_t can_number) {
  // FDCAN1 = trans 1, FDCAN3 = trans 3, FDCAN2 = trans 2 normal or 4 flipped harness
  uint8_t transceiver_number = can_number;
  if (can_number == 2U) {
    uint8_t flip = (car_harness_status == HARNESS_STATUS_FLIPPED) ? 2U : 0U;
    transceiver_number += flip;
  }
  current_board->enable_can_transceiver(transceiver_number, false);
  delay(20000);
  current_board->enable_can_transceiver(transceiver_number, true);
  bus_off_err[can_number] = 0U;
  puts("Cycled transceiver number: "); puth(transceiver_number); puts("\n");
}

// ***************************** CAN *****************************
void process_can(uint8_t can_number) {
  if (can_number != 0xffU) {
    ENTER_CRITICAL();

    FDCAN_GlobalTypeDef *CANx = CANIF_FROM_CAN_NUM(can_number);
    uint8_t bus_number = BUS_NUM_FROM_CAN_NUM(can_number);
    
    CANx->IR |= FDCAN_IR_TFE; // Clear Tx FIFO Empty flag

    if ((CANx->TXFQS & FDCAN_TXFQS_TFQF) == 0) {
      CAN_FIFOMailBox_TypeDef to_send;
      if (can_pop(can_queues[bus_number], &to_send)) {
        can_tx_cnt += 1;
        uint32_t TxFIFOSA = FDCAN_START_ADDRESS + (can_number * FDCAN_OFFSET) + (FDCAN_RX_FIFO_0_EL_CNT * FDCAN_RX_FIFO_0_EL_SIZE);
        uint8_t tx_index = (CANx->TXFQS >> FDCAN_TXFQS_TFQPI_Pos) & 0x1F;
        // only send if we have received a packet
        CAN_FIFOMailBox_TypeDef *fifo;
        fifo = (CAN_FIFOMailBox_TypeDef *)(TxFIFOSA + (tx_index * FDCAN_TX_FIFO_EL_SIZE));

        // Convert from "mailbox type"
        fifo->RIR = ((to_send.RIR & 0x6) << 28) | (to_send.RIR >> 3);  // identifier format and frame type | identifier
        //REDEBUG: enable CAN FD and BRS for test purposes
        //fifo->RDTR = ((to_send.RDTR & 0xF) << 16) | ((to_send.RDTR) >> 16) | (1U << 21) | (1U << 20); // DLC (length) | timestamp | enable CAN FD | enable BRS
        fifo->RDTR = ((to_send.RDTR & 0xF) << 16) | ((to_send.RDTR) >> 16); // DLC (length) | timestamp
        fifo->RDLR = to_send.RDLR;
        fifo->RDHR = to_send.RDHR;
        
        CANx->TXBAR = (1UL << tx_index); 

        // Send back to USB
        can_txd_cnt += 1;
        CAN_FIFOMailBox_TypeDef to_push;
        to_push.RIR = to_send.RIR;
        to_push.RDTR = (to_send.RDTR & 0xFFFF000FU) | ((CAN_BUS_RET_FLAG | bus_number) << 4);
        to_push.RDLR = to_send.RDLR;
        to_push.RDHR = to_send.RDHR;
        can_send_errs += can_push(&can_rx_q, &to_push) ? 0U : 1U;

        usb_cb_ep3_out_complete();
      }
    }

    // Recover after Bus-off state
    if (((CANx->PSR & FDCAN_PSR_BO) != 0) && ((CANx->CCCR & FDCAN_CCCR_INIT) != 0)) {
      bus_off_err[can_number] += 1U;
      puts("CAN is in Bus_Off state! Resetting... CAN number: "); puth(can_number); puts("\n");
      if (bus_off_err[can_number] > BUS_OFF_FAIL_LIMIT) {
        cycle_transceiver(can_number);
      }
      CANx->IR = 0xFFC60000U; // Reset all flags(Only errors!)
      CANx->CCCR &= ~(FDCAN_CCCR_INIT);
      uint32_t timeout_counter = 0U;
      while((CANx->CCCR & FDCAN_CCCR_INIT) != 0) {
        // Delay for about 1ms
        delay(10000);
        timeout_counter++;

        if(timeout_counter >= CAN_INIT_TIMEOUT_MS){
          puts(CAN_NAME_FROM_CANIF(CANx)); puts(" Bus_Off reset timed out!\n");
          break;
        }
      }
    }
    EXIT_CRITICAL();
  }
}

// CAN receive handlers
// blink blue when we are receiving CAN messages
void can_rx(uint8_t can_number) {
  FDCAN_GlobalTypeDef *CANx = CANIF_FROM_CAN_NUM(can_number);
  uint8_t bus_number = BUS_NUM_FROM_CAN_NUM(can_number);
  uint8_t rx_fifo_idx;

  // Rx FIFO 0 new message
  if((CANx->IR & FDCAN_IR_RF0N) != 0) {
    CANx->IR |= FDCAN_IR_RF0N;
    while((CANx->RXF0S & FDCAN_RXF0S_F0FL) != 0) {
      can_rx_cnt += 1;

      // can is live
      pending_can_live = 1;

      // getting new message index (0 to 63)
      rx_fifo_idx = (uint8_t)((CANx->RXF0S >> FDCAN_RXF0S_F0GI_Pos) & 0x3F);

      uint32_t RxFIFO0SA = FDCAN_START_ADDRESS + (can_number * FDCAN_OFFSET);
      CAN_FIFOMailBox_TypeDef to_push;
      CAN_FIFOMailBox_TypeDef *fifo;

      // getting address
      fifo = (CAN_FIFOMailBox_TypeDef *)(RxFIFO0SA + (rx_fifo_idx * FDCAN_RX_FIFO_0_EL_SIZE));

      // Need to convert real CAN frame format to mailbox "type"
      to_push.RIR = ((fifo->RIR >> 28) & 0x6) | (fifo->RIR << 3); // identifier format and frame type | identifier
      to_push.RDTR = ((fifo->RDTR >> 16) & 0xF) | (fifo->RDTR << 16); // DLC (length) | timestamp
      to_push.RDLR = fifo->RDLR;
      to_push.RDHR = fifo->RDHR;

      // modify RDTR for our API
      to_push.RDTR = (to_push.RDTR & 0xFFFF000F) | (bus_number << 4);

      // forwarding (panda only)
      int bus_fwd_num = (can_forwarding[bus_number] != -1) ? can_forwarding[bus_number] : safety_fwd_hook(bus_number, &to_push);
      if (bus_fwd_num != -1) {
        CAN_FIFOMailBox_TypeDef to_send;
        to_send.RIR = to_push.RIR;
        to_send.RDTR = to_push.RDTR;
        to_send.RDLR = to_push.RDLR;
        to_send.RDHR = to_push.RDHR;
        can_send(&to_send, bus_fwd_num, true);
      }

      can_rx_errs += safety_rx_hook(&to_push) ? 0U : 1U;
      ignition_can_hook(&to_push);

      current_board->set_led(LED_BLUE, true);
      can_send_errs += can_push(&can_rx_q, &to_push) ? 0U : 1U;

      // update read index 
      CANx->RXF0A = rx_fifo_idx;
    }

  } else if((CANx->IR & (FDCAN_IR_PEA | FDCAN_IR_PED | FDCAN_IR_RF0L | FDCAN_IR_RF0F | FDCAN_IR_EW | FDCAN_IR_MRAF | FDCAN_IR_TOO)) != 0) {
    #ifdef DEBUG
      puts("FDCAN error, FDCAN_IR: ");puth(CANx->IR);puts("\n");
    #endif
    CANx->IR |= (FDCAN_IR_PEA | FDCAN_IR_PED | FDCAN_IR_RF0L | FDCAN_IR_RF0F | FDCAN_IR_EW | FDCAN_IR_MRAF | FDCAN_IR_TOO); // Clean all error flags
    can_err_cnt += 1;
  } else { 
    
  }
  
}

void FDCAN1_IT0_IRQ_Handler(void) { can_rx(0); }
void FDCAN1_IT1_IRQ_Handler(void) { process_can(0); }

void FDCAN2_IT0_IRQ_Handler(void) { can_rx(1); }
void FDCAN2_IT1_IRQ_Handler(void) { process_can(1); }

void FDCAN3_IT0_IRQ_Handler(void) { can_rx(2);  }
void FDCAN3_IT1_IRQ_Handler(void) { process_can(2); }

bool can_init(uint8_t can_number) {
  bool ret = false;

  REGISTER_INTERRUPT(FDCAN1_IT0_IRQn, FDCAN1_IT0_IRQ_Handler, CAN_INTERRUPT_RATE, FAULT_INTERRUPT_RATE_CAN_1)
  REGISTER_INTERRUPT(FDCAN1_IT1_IRQn, FDCAN1_IT1_IRQ_Handler, CAN_INTERRUPT_RATE, FAULT_INTERRUPT_RATE_CAN_1)
  REGISTER_INTERRUPT(FDCAN2_IT0_IRQn, FDCAN2_IT0_IRQ_Handler, CAN_INTERRUPT_RATE, FAULT_INTERRUPT_RATE_CAN_2)
  REGISTER_INTERRUPT(FDCAN2_IT1_IRQn, FDCAN2_IT1_IRQ_Handler, CAN_INTERRUPT_RATE, FAULT_INTERRUPT_RATE_CAN_2)
  REGISTER_INTERRUPT(FDCAN3_IT0_IRQn, FDCAN3_IT0_IRQ_Handler, CAN_INTERRUPT_RATE, FAULT_INTERRUPT_RATE_CAN_3)
  REGISTER_INTERRUPT(FDCAN3_IT1_IRQn, FDCAN3_IT1_IRQ_Handler, CAN_INTERRUPT_RATE, FAULT_INTERRUPT_RATE_CAN_3)

  if (can_number != 0xffU) {
    FDCAN_GlobalTypeDef *CANx = CANIF_FROM_CAN_NUM(can_number);
    ret &= can_set_speed(can_number);
    ret &= llcan_init(CANx);
    // in case there are queued up messages
    process_can(can_number);
  }
  return ret;
}
