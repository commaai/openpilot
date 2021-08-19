typedef struct {
  volatile uint32_t w_ptr;
  volatile uint32_t r_ptr;
  uint32_t fifo_size;
  CAN_FIFOMailBox_TypeDef *elems;
} can_ring;

#define CAN_BUS_RET_FLAG 0x80U
#define CAN_BUS_NUM_MASK 0x7FU

#define BUS_MAX 4U

uint32_t can_rx_errs = 0;
uint32_t can_send_errs = 0;
uint32_t can_fwd_errs = 0;
uint32_t gmlan_send_errs = 0;

extern int can_live;
extern int pending_can_live;

// must reinit after changing these
extern int can_loopback;
extern int can_silent;
extern uint32_t can_speed[4];
extern uint32_t can_data_speed[3];

// Ignition detected from CAN meessages
bool ignition_can = false;
bool ignition_cadillac = false;
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
  CAN_FIFOMailBox_TypeDef elems_##x[size]; \
  can_ring can_##x = { .w_ptr = 0, .r_ptr = 0, .fifo_size = (size), .elems = (CAN_FIFOMailBox_TypeDef *)&(elems_##x) };

can_buffer(rx_q, 0x1000)
can_buffer(tx1_q, 0x100)
can_buffer(tx2_q, 0x100)
can_buffer(tx3_q, 0x100)
can_buffer(txgmlan_q, 0x100)
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
bool can_pop(can_ring *q, CAN_FIFOMailBox_TypeDef *elem) {
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

bool can_push(can_ring *q, CAN_FIFOMailBox_TypeDef *elem) {
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
      puts("can_push failed!\n");
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
}

// assign CAN numbering
// bus num: Can bus number on ODB connector. Sent to/from USB
//    Min: 0; Max: 127; Bit 7 marks message as receipt (bus 129 is receipt for but 1)
// cans: Look up MCU can interface from bus number
// can number: numeric lookup for MCU CAN interfaces (0 = CAN1, 1 = CAN2, etc);
// bus_lookup: Translates from 'can number' to 'bus number'.
// can_num_lookup: Translates from 'bus number' to 'can number'.
// can_forwarding: Given a bus num, lookup bus num to forward to. -1 means no forward.

// Helpers
// Panda:       Bus 0=CAN1   Bus 1=CAN2   Bus 2=CAN3
uint8_t bus_lookup[] = {0,1,2};
uint8_t can_num_lookup[] = {0,1,2,-1};
int8_t can_forwarding[] = {-1,-1,-1,-1};
uint32_t can_speed[] = {5000, 5000, 5000, 333};
uint32_t can_data_speed[] = {5000, 5000, 5000}; //For CAN FD with BRS only
#define CAN_MAX 3U

#define CANIF_FROM_CAN_NUM(num) (cans[num])
#define BUS_NUM_FROM_CAN_NUM(num) (bus_lookup[num])
#define CAN_NUM_FROM_BUS_NUM(num) (can_num_lookup[num])

void can_init_all(void) {
  bool ret = true;
  for (uint8_t i=0U; i < CAN_MAX; i++) {
    can_clear(can_queues[i]);
    ret &= can_init(i);
  }
  UNUSED(ret);
}

void can_flip_buses(uint8_t bus1, uint8_t bus2){
  bus_lookup[bus1] = bus2;
  bus_lookup[bus2] = bus1;
  can_num_lookup[bus1] = bus2;
  can_num_lookup[bus2] = bus1;
}

void ignition_can_hook(CAN_FIFOMailBox_TypeDef *to_push) {
  int bus = GET_BUS(to_push);
  int addr = GET_ADDR(to_push);
  int len = GET_LEN(to_push);

  ignition_can_cnt = 0U;  // reset counter

  if (bus == 0) {
    // TODO: verify on all supported GM models that we can reliably detect ignition using only this signal,
    // since the 0x1F1 signal can briefly go low immediately after ignition
    if ((addr == 0x160) && (len == 5)) {
      // this message isn't all zeros when ignition is on
      ignition_cadillac = GET_BYTES_04(to_push) != 0;
    }
    // GM exception
    if ((addr == 0x1F1) && (len == 8)) {
      // Bit 5 is ignition "on"
      bool ignition_gm = ((GET_BYTE(to_push, 0) & 0x20) != 0);
      ignition_can = ignition_gm || ignition_cadillac;
    }
    // Tesla exception
    if ((addr == 0x348) && (len == 8)) {
      // GTW_status
      ignition_can = (GET_BYTE(to_push, 0) & 0x1) != 0;
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

void can_send(CAN_FIFOMailBox_TypeDef *to_push, uint8_t bus_number, bool skip_tx_hook) {
  if (skip_tx_hook || safety_tx_hook(to_push) != 0) {
    if (bus_number < BUS_MAX) {
      // add CAN packet to send queue
      // bus number isn't passed through
      to_push->RDTR &= 0xF;
      if ((bus_number == 3U) && (can_num_lookup[3] == 0xFFU)) {
        gmlan_send_errs += bitbang_gmlan(to_push) ? 0U : 1U;
      } else {
        can_fwd_errs += can_push(can_queues[bus_number], to_push) ? 0U : 1U;
        process_can(CAN_NUM_FROM_BUS_NUM(bus_number));
      }
    }
  }
}

void can_set_forwarding(int from, int to) {
  can_forwarding[from] = to;
}
