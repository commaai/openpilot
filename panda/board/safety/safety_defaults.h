bool HKG_forward_BUS1 = false;
bool HKG_forward_BUS2 = true;
bool HKG_LCAN_on_BUS1 = false;
int HKG_MDPS12_checksum = -1;
int HKG_MDPS12_cnt = 0;   
int HKG_last_StrColT = 0;

int default_rx_hook(CAN_FIFOMailBox_TypeDef *to_push) {
  int bus = GET_BUS(to_push);
  int addr = GET_ADDR(to_push);

  // check if LKAS connected to Bus0
  if ((addr == 832) && (bus == 0)) {
    if (HKG_forward_BUS2 != false) {
      HKG_forward_BUS2 = false;
	}
  }
  // check if we have a LCAN on Bus1
  if (bus == 1 && addr == 1296) {
    HKG_LCAN_on_BUS1 = true;
  }
  // check if we have a MDPS or SCC on Bus1
  if (bus == 1 && (addr == 593 || addr == 897 || addr == 1057) && !HKG_LCAN_on_BUS1) {
    if (HKG_forward_BUS1 != true) {
      HKG_forward_BUS1 = true;
	}
  }

  if ((addr == 593) && (HKG_MDPS12_checksum == -1)){
    int New_Chksum2 = 0;
    uint8_t dat[8];
    for (int i=0; i<8; i++) {
      dat[i] = GET_BYTE(to_push, i);
    }
    int Chksum2 = dat[3];
    dat[3] = 0;
    for (int i=0; i<8; i++) {
      New_Chksum2 += dat[i];
    }
    New_Chksum2 %= 256;
    if (Chksum2 == New_Chksum2) {
      HKG_MDPS12_checksum = 0;
    }
    else {
      HKG_MDPS12_checksum = 1;
    }
  }
  return true;
}

// *** no output safety mode ***

static void nooutput_init(int16_t param) {
  UNUSED(param);
  controls_allowed = false;
  relay_malfunction_reset();
}

static int nooutput_tx_hook(CAN_FIFOMailBox_TypeDef *to_send) {
  UNUSED(to_send);
  return false;
}

static int nooutput_tx_lin_hook(int lin_num, uint8_t *data, int len) {
  UNUSED(lin_num);
  UNUSED(data);
  UNUSED(len);
  return false;
}

static int default_fwd_hook(int bus_num, CAN_FIFOMailBox_TypeDef *to_fwd) {
  int addr = GET_ADDR(to_fwd);
  int bus_fwd = -1;
  int HKG_bus1 = 0, HKG_bus2 = 0;
  if (HKG_forward_BUS1){HKG_bus1 = 1;}
  if (HKG_forward_BUS2){HKG_bus2 = 2;}

  if (bus_num == 0 && (HKG_forward_BUS1 || HKG_forward_BUS2)) {
    bus_fwd = (HKG_forward_BUS1 && HKG_forward_BUS2) ? 12 : HKG_bus1 + HKG_bus2;
  }
  if (bus_num == 1 && HKG_forward_BUS1) {
    bus_fwd = HKG_bus2 * 10;
  }
  if (bus_num == 2 && HKG_forward_BUS2) {
    bus_fwd = HKG_bus1 * 10;
  }
    // Code for LKA/LFA/HDA anti-nagging.
  if (addr == 593 && bus_fwd != -1) {
    uint8_t dat[8];
    int New_Chksum2 = 0;
    for (int i=0; i<8; i++) {
      dat[i] = GET_BYTE(to_fwd, i);
    }
    if (HKG_MDPS12_cnt > 330) {
      int StrColTq = dat[0] | (dat[1] & 0x7) << 8;
      int OutTq = dat[6] >> 4 | dat[7] << 4;
      if (HKG_MDPS12_cnt == 331) {
        StrColTq -= 164;
      } else {
        StrColTq = HKG_last_StrColT + 34;
      }
      OutTq = 2058;
      dat[0] = StrColTq & 0xFF;
      dat[1] &= 0xF8;
      dat[1] |= StrColTq >> 8;
      dat[6] &= 0xF;
      dat[6] |= (OutTq & 0xF) << 4;
      dat[7] = OutTq >> 4;
      to_fwd->RDLR &= 0xFFF800;
      to_fwd->RDLR |= StrColTq;
      to_fwd->RDHR &= 0xFFFFF;
      to_fwd->RDHR |= OutTq << 20;
      HKG_last_StrColT = StrColTq;
      dat[3] = 0;
      if (!HKG_MDPS12_checksum) { 
        for (int i=0; i<8; i++) {
          New_Chksum2 += dat[i];
        }
        New_Chksum2 %= 256;
      } else if (HKG_MDPS12_checksum) {
        uint8_t crc = 0xFF;
        uint8_t poly = 0x1D;
        int i, j;
        for (i=0; i<8; i++){
          if (i!=3){ //don't include CRC byte
            crc ^= dat[i];
            for (j=0; j<8; j++) {
              if ((crc & 0x80) != 0U) {
                crc = (crc << 1) ^ poly;
              } else {
                crc <<= 1;
              }
            }
          }
        }
        crc ^= 0xFF;
        crc %= 256;
        New_Chksum2 = crc;
      }
      to_fwd->RDLR |= New_Chksum2 << 24;
    }
    HKG_MDPS12_cnt += 1;
    HKG_MDPS12_cnt %= 345;
  }
  return bus_fwd;
}

const safety_hooks nooutput_hooks = {
  .init = nooutput_init,
  .rx = default_rx_hook,
  .tx = nooutput_tx_hook,
  .tx_lin = nooutput_tx_lin_hook,
  .fwd = default_fwd_hook,
};

// *** all output safety mode ***

static void alloutput_init(int16_t param) {
  UNUSED(param);
  controls_allowed = true;
  relay_malfunction_reset();
}

static int alloutput_tx_hook(CAN_FIFOMailBox_TypeDef *to_send) {
  UNUSED(to_send);
  return true;
}

static int alloutput_tx_lin_hook(int lin_num, uint8_t *data, int len) {
  UNUSED(lin_num);
  UNUSED(data);
  UNUSED(len);
  return true;
}

const safety_hooks alloutput_hooks = {
  .init = alloutput_init,
  .rx = default_rx_hook,
  .tx = alloutput_tx_hook,
  .tx_lin = alloutput_tx_lin_hook,
  .fwd = default_fwd_hook,
};
