typedef struct {
  volatile uint32_t w_ptr;
  volatile uint32_t r_ptr;
  uint32_t fifo_size;
  CANPacket_t *elems;
} can_ring;

typedef struct {
  uint8_t bus_lookup;
  uint8_t can_num_lookup;
  uint32_t can_speed;
  uint32_t can_data_speed;
  bool canfd_enabled;
  bool brs_enabled;
} bus_config_t;

uint32_t can_rx_errs = 0;
uint32_t can_send_errs = 0;
uint32_t can_fwd_errs = 0;
uint32_t gmlan_send_errs = 0;
uint32_t blocked_msg_cnt = 0;

extern int can_live;
extern int pending_can_live;

// must reinit after changing these
extern int can_loopback;
extern int can_silent;

// Ignition detected from CAN meessages
bool ignition_can = false;
uint32_t ignition_can_cnt = 0U;

#define ALL_CAN_SILENT 0xFF
#define ALL_CAN_LIVE 0

int can_live = 0;
int pending_can_live = 0;
int can_loopback = 0;
int can_silent = ALL_CAN_SILENT;

// ******************* functions prototypes *********************
bool can_init(uint8_t can_number);
void process_can(uint8_t can_number);

// ********************* instantiate queues *********************
#define can_buffer(x, size) \
  CANPacket_t elems_##x[size]; \
  can_ring can_##x = { .w_ptr = 0, .r_ptr = 0, .fifo_size = (size), .elems = (CANPacket_t *)&(elems_##x) };

#ifdef STM32H7
__attribute__((section(".ram_d1"))) can_buffer(rx_q, 0x1000)
__attribute__((section(".ram_d1"))) can_buffer(txgmlan_q, 0x1A0)
#else
can_buffer(rx_q, 0x1000)
can_buffer(txgmlan_q, 0x1A0)
#endif
can_buffer(tx1_q, 0x1A0)
can_buffer(tx2_q, 0x1A0)
can_buffer(tx3_q, 0x1A0)
// FIXME:
// cppcheck-suppress misra-c2012-9.3
can_ring *can_queues[] = {&can_tx1_q, &can_tx2_q, &can_tx3_q, &can_txgmlan_q};

// global CAN stats
int can_rx_cnt = 0;
int can_tx_cnt = 0;
int can_txd_cnt = 0;
int can_err_cnt = 0;
int can_overflow_cnt = 0;

// ********************* interrupt safe queue *********************
bool can_pop(can_ring *q, CANPacket_t *elem) {
  bool ret = 0;

  ENTER_CRITICAL();
  if (q->w_ptr != q->r_ptr) {
    *elem = q->elems[q->r_ptr];
    if ((q->r_ptr + 1U) == q->fifo_size) {
      q->r_ptr = 0;
    } else {
      q->r_ptr += 1U;
    }
    ret = 1;
  }
  EXIT_CRITICAL();

  return ret;
}

bool can_push(can_ring *q, CANPacket_t *elem) {
  bool ret = false;
  uint32_t next_w_ptr;

  ENTER_CRITICAL();
  if ((q->w_ptr + 1U) == q->fifo_size) {
    next_w_ptr = 0;
  } else {
    next_w_ptr = q->w_ptr + 1U;
  }
  if (next_w_ptr != q->r_ptr) {
    q->elems[q->w_ptr] = *elem;
    q->w_ptr = next_w_ptr;
    ret = true;
  }
  EXIT_CRITICAL();
  if (!ret) {
    can_overflow_cnt++;
    #ifdef DEBUG
      puts("can_push to ");
      if (q == &can_rx_q) {
        puts("can_rx_q");
      } else if (q == &can_tx1_q) {
        puts("can_tx1_q");
      } else if (q == &can_tx2_q) {
        puts("can_tx2_q");
      } else if (q == &can_tx3_q) {
        puts("can_tx3_q");
      } else if (q == &can_txgmlan_q) {
        puts("can_txgmlan_q");
      } else {
        puts("unknown");
      }
      puts(" failed!\n");
    #endif
  }
  return ret;
}

uint32_t can_slots_empty(can_ring *q) {
  uint32_t ret = 0;

  ENTER_CRITICAL();
  if (q->w_ptr >= q->r_ptr) {
    ret = q->fifo_size - 1U - q->w_ptr + q->r_ptr;
  } else {
    ret = q->r_ptr - q->w_ptr - 1U;
  }
  EXIT_CRITICAL();

  return ret;
}

void can_clear(can_ring *q) {
  ENTER_CRITICAL();
  q->w_ptr = 0;
  q->r_ptr = 0;
  EXIT_CRITICAL();
  // handle TX buffer full with zero ECUs awake on the bus
  usb_cb_ep3_out_complete();
}

// assign CAN numbering
// bus num: Can bus number on ODB connector. Sent to/from USB
//    Min: 0; Max: 127; Bit 7 marks message as receipt (bus 129 is receipt for but 1)
// cans: Look up MCU can interface from bus number
// can number: numeric lookup for MCU CAN interfaces (0 = CAN1, 1 = CAN2, etc);
// bus_lookup: Translates from 'can number' to 'bus number'.
// can_num_lookup: Translates from 'bus number' to 'can number'.

// Helpers
// Panda:       Bus 0=CAN1   Bus 1=CAN2   Bus 2=CAN3
bus_config_t bus_config[] = {
  { .bus_lookup = 0U, .can_num_lookup = 0U, .can_speed = 5000U, .can_data_speed = 20000U, .canfd_enabled = false, .brs_enabled = false },
  { .bus_lookup = 1U, .can_num_lookup = 1U, .can_speed = 5000U, .can_data_speed = 20000U, .canfd_enabled = false, .brs_enabled = false },
  { .bus_lookup = 2U, .can_num_lookup = 2U, .can_speed = 5000U, .can_data_speed = 20000U, .canfd_enabled = false, .brs_enabled = false },
  { .bus_lookup = 0xFFU, .can_num_lookup = 0xFFU, .can_speed = 333U, .can_data_speed = 333U, .canfd_enabled = false, .brs_enabled = false },
};

#define CANIF_FROM_CAN_NUM(num) (cans[num])
#define BUS_NUM_FROM_CAN_NUM(num) (bus_config[num].bus_lookup)
#define CAN_NUM_FROM_BUS_NUM(num) (bus_config[num].can_num_lookup)

void can_init_all(void) {
  bool ret = true;
  for (uint8_t i=0U; i < CAN_CNT; i++) {
    can_clear(can_queues[i]);
    ret &= can_init(i);
  }
  UNUSED(ret);
}

void can_flip_buses(uint8_t bus1, uint8_t bus2){
  bus_config[bus1].bus_lookup = bus2;
  bus_config[bus2].bus_lookup = bus1;
  bus_config[bus1].can_num_lookup = bus2;
  bus_config[bus2].can_num_lookup = bus1;
}

void ignition_can_hook(CANPacket_t *to_push) {
  int bus = GET_BUS(to_push);
  int addr = GET_ADDR(to_push);
  int len = GET_LEN(to_push);

  ignition_can_cnt = 0U;  // reset counter

  if (bus == 0) {
    // GM exception
    if ((addr == 0x160) && (len == 5)) {
      // this message isn't all zeros when ignition is on
      ignition_can = GET_BYTES_04(to_push) != 0U;
    }

    // Tesla exception
    if ((addr == 0x348) && (len == 8)) {
      // GTW_status
      ignition_can = (GET_BYTE(to_push, 0) & 0x1U) != 0U;
    }

    // Mazda exception
    if ((addr == 0x9E) && (len == 8)) {
      ignition_can = (GET_BYTE(to_push, 0) >> 5) == 0x6U;
    }

  }
}

bool can_tx_check_min_slots_free(uint32_t min) {
  return
    (can_slots_empty(&can_tx1_q) >= min) &&
    (can_slots_empty(&can_tx2_q) >= min) &&
    (can_slots_empty(&can_tx3_q) >= min) &&
    (can_slots_empty(&can_txgmlan_q) >= min);
}

void can_send(CANPacket_t *to_push, uint8_t bus_number, bool skip_tx_hook) {
  if (skip_tx_hook || safety_tx_hook(to_push) != 0) {
    if (bus_number < BUS_CNT) {
      // add CAN packet to send queue
      if ((bus_number == 3U) && (bus_config[3].can_num_lookup == 0xFFU)) {
        gmlan_send_errs += bitbang_gmlan(to_push) ? 0U : 1U;
      } else {
        can_fwd_errs += can_push(can_queues[bus_number], to_push) ? 0U : 1U;
        process_can(CAN_NUM_FROM_BUS_NUM(bus_number));
      }
    }
  } else {
    blocked_msg_cnt += 1U;
    to_push->rejected = 1U;
    can_send_errs += can_push(&can_rx_q, to_push) ? 0U : 1U;
  }
}
