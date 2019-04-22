int tesla_radar_status = 0; //0-not present, 1-initializing, 2-active
uint32_t tesla_last_radar_signal = 0;
const int TESLA_RADAR_TIMEOUT = 1000000; // 1 second between real time checks
char radar_VIN[] = "5YJSA1H27FF087536"; //leave empty if your radar VIN matches the car VIN
int tesla_radar_can = 1; // 0, 1 or 2 set from EON via fake message
int tesla_radar_vin_complete = 0; //set to 7 when complete vin is received
int tesla_radar_should_send = 0; //set to 1 from EON via fake message when we want to use it
int tesla_radar_counter = 0; //counter to determine when to send messages
int tesla_radar_trigger_message_id = 0; //id of the message at 100Hz to trigger the radar data
int actual_speed_kph = 0; //use the rx_hook to set this to the car speed in kph; used by radar
int tesla_radar_config_message_id = 0x560; //message used to send VIN to Panda

//message IDs and counters
int tesla_radar_x2B9_id = 0;
int tesla_radar_x159_id = 0;
int tesla_radar_x219_id = 0;
int tesla_radar_x149_id = 0;
int tesla_radar_x129_id = 0;
int tesla_radar_x1A9_id = 0;
int tesla_radar_x199_id = 0;
int tesla_radar_x169_id = 0;
int tesla_radar_x119_id = 0;
int tesla_radar_x109_id = 0;

static int add_tesla_crc(uint32_t MLB, uint32_t MHB , int msg_len) {
  //"""Calculate CRC8 using 1D poly, FF start, FF end"""
  int crc_lookup[256] = { 0x00, 0x1D, 0x3A, 0x27, 0x74, 0x69, 0x4E, 0x53, 0xE8, 0xF5, 0xD2, 0xCF, 0x9C, 0x81, 0xA6, 0xBB, 
    0xCD, 0xD0, 0xF7, 0xEA, 0xB9, 0xA4, 0x83, 0x9E, 0x25, 0x38, 0x1F, 0x02, 0x51, 0x4C, 0x6B, 0x76, 
    0x87, 0x9A, 0xBD, 0xA0, 0xF3, 0xEE, 0xC9, 0xD4, 0x6F, 0x72, 0x55, 0x48, 0x1B, 0x06, 0x21, 0x3C, 
    0x4A, 0x57, 0x70, 0x6D, 0x3E, 0x23, 0x04, 0x19, 0xA2, 0xBF, 0x98, 0x85, 0xD6, 0xCB, 0xEC, 0xF1, 
    0x13, 0x0E, 0x29, 0x34, 0x67, 0x7A, 0x5D, 0x40, 0xFB, 0xE6, 0xC1, 0xDC, 0x8F, 0x92, 0xB5, 0xA8, 
    0xDE, 0xC3, 0xE4, 0xF9, 0xAA, 0xB7, 0x90, 0x8D, 0x36, 0x2B, 0x0C, 0x11, 0x42, 0x5F, 0x78, 0x65, 
    0x94, 0x89, 0xAE, 0xB3, 0xE0, 0xFD, 0xDA, 0xC7, 0x7C, 0x61, 0x46, 0x5B, 0x08, 0x15, 0x32, 0x2F, 
    0x59, 0x44, 0x63, 0x7E, 0x2D, 0x30, 0x17, 0x0A, 0xB1, 0xAC, 0x8B, 0x96, 0xC5, 0xD8, 0xFF, 0xE2, 
    0x26, 0x3B, 0x1C, 0x01, 0x52, 0x4F, 0x68, 0x75, 0xCE, 0xD3, 0xF4, 0xE9, 0xBA, 0xA7, 0x80, 0x9D, 
    0xEB, 0xF6, 0xD1, 0xCC, 0x9F, 0x82, 0xA5, 0xB8, 0x03, 0x1E, 0x39, 0x24, 0x77, 0x6A, 0x4D, 0x50, 
    0xA1, 0xBC, 0x9B, 0x86, 0xD5, 0xC8, 0xEF, 0xF2, 0x49, 0x54, 0x73, 0x6E, 0x3D, 0x20, 0x07, 0x1A, 
    0x6C, 0x71, 0x56, 0x4B, 0x18, 0x05, 0x22, 0x3F, 0x84, 0x99, 0xBE, 0xA3, 0xF0, 0xED, 0xCA, 0xD7, 
    0x35, 0x28, 0x0F, 0x12, 0x41, 0x5C, 0x7B, 0x66, 0xDD, 0xC0, 0xE7, 0xFA, 0xA9, 0xB4, 0x93, 0x8E, 
    0xF8, 0xE5, 0xC2, 0xDF, 0x8C, 0x91, 0xB6, 0xAB, 0x10, 0x0D, 0x2A, 0x37, 0x64, 0x79, 0x5E, 0x43, 
    0xB2, 0xAF, 0x88, 0x95, 0xC6, 0xDB, 0xFC, 0xE1, 0x5A, 0x47, 0x60, 0x7D, 0x2E, 0x33, 0x14, 0x09, 
    0x7F, 0x62, 0x45, 0x58, 0x0B, 0x16, 0x31, 0x2C, 0x97, 0x8A, 0xAD, 0xB0, 0xE3, 0xFE, 0xD9, 0xC4 };
  int crc = 0xFF;
  for (int x = 0; x < msg_len; x++) {
    int v = 0;
    if (x <= 3) {
      v = (MLB >> (x * 8)) & 0xFF;
    } else {
      v = (MHB >> ( (x-4) * 8)) & 0xFF;
    }
    crc = crc_lookup[crc ^ v];
  }
  crc = crc ^ 0xFF;
  return crc;
}

static int add_tesla_cksm(CAN_FIFOMailBox_TypeDef *msg , int msg_id, int msg_len) {
  int cksm = (0xFF & msg_id) + (0xFF & (msg_id >> 8));
  for (int x = 0; x < msg_len; x++) {
    int v = 0;
    if (x <= 3) {
      v = (msg->RDLR >> (x * 8)) & 0xFF;
    } else {
      v = (msg->RDHR >> ( (x-4) * 8)) & 0xFF;
    }
    cksm = (cksm + v) & 0xFF;
  }
  return cksm;
}

static int add_tesla_cksm2(uint32_t dl, uint32_t dh, int msg_id, int msg_len) {
  CAN_FIFOMailBox_TypeDef to_check;
  to_check.RDLR = dl;
  to_check.RDHR = dh;
  return add_tesla_cksm(&to_check,msg_id,msg_len);
}

static void send_fake_message(uint32_t RIR, uint32_t RDTR,int msg_len, int msg_addr, int bus_num, uint32_t data_lo, uint32_t data_hi) {
  CAN_FIFOMailBox_TypeDef to_send;
  uint32_t addr_mask = 0x001FFFFF;
  to_send.RIR = (msg_addr << 21) + (addr_mask & (RIR | 1));
  to_send.RDTR = (RDTR & 0xFFFFFFF0) | msg_len;
  to_send.RDLR = data_lo;
  to_send.RDHR = data_hi;
  can_send(&to_send, bus_num);
}

static uint32_t radar_VIN_char(int pos, int shift) {
  return (((int)radar_VIN[pos]) << (shift * 8));
}

static void activate_tesla_radar(uint32_t RIR, uint32_t RDTR) {
    //if we did not receive the VIN or no request to activate radar, then return
    if ((tesla_radar_vin_complete != 7) || (tesla_radar_should_send == 0)) {
        return;
    }
    uint32_t MLB;
    uint32_t MHB;
    //send all messages at 100Hz
    //send 199
    MLB = 0x00207D2F;
    MHB = 0x0000FF04 + (tesla_radar_x199_id << 20);
    int crc = add_tesla_crc(MLB, MHB,7);
    MHB = MHB +(crc << 24);
    tesla_radar_x199_id++;
    tesla_radar_x199_id = tesla_radar_x199_id % 16;
    send_fake_message(RIR,RDTR,8,0x199,tesla_radar_can,MLB,MHB);
    //send 169
    int speed_kph = (int)(actual_speed_kph/0.04) & 0x1FFF;
    MLB = (speed_kph | (speed_kph << 13) | (speed_kph << 26)) & 0xFFFFFFFF;
    MHB = ((speed_kph  >> 6) | (speed_kph << 7) | (tesla_radar_x169_id << 20)) & 0x00FFFFFF;
    int cksm = add_tesla_cksm2(MLB, MHB, 0x76, 7);
    MHB = MHB + (cksm << 24);
    tesla_radar_x169_id++;
    tesla_radar_x169_id = tesla_radar_x169_id % 16;
    send_fake_message(RIR,RDTR,8,0x169,tesla_radar_can,MLB,MHB);
    //send 119
    MLB = 0x11F41FFF;
    MHB = 0x00000080 + tesla_radar_x119_id;
    cksm = add_tesla_cksm2(MLB, MHB, 0x17, 5);
    MHB = MHB + (cksm << 8);
    tesla_radar_x119_id++;
    tesla_radar_x119_id = tesla_radar_x119_id % 16;
    send_fake_message(RIR,RDTR,6,0x119,tesla_radar_can,MLB,MHB);
    //send 109
    MLB = 0x80000000 + (tesla_radar_x109_id << 13);
    MHB = 0x00; 
    cksm = add_tesla_cksm2(MLB, MHB, 0x7, 7);
    MHB = MHB + (cksm << 24);
    tesla_radar_x109_id++;
    tesla_radar_x109_id = tesla_radar_x109_id % 8;
    send_fake_message(RIR,RDTR,8,0x109,tesla_radar_can,MLB,MHB);
    //send all messages at 50Hz
    if (tesla_radar_counter % 2 ==0) {
        //send 159
        MLB = 0x0B4FFFFB + (tesla_radar_x159_id << 28);
        MHB = 0x000000FF;
        cksm = add_tesla_cksm2(MLB, MHB, 0xB2, 5);
        MHB = MHB +(cksm << 8);
        tesla_radar_x159_id++;
        tesla_radar_x159_id = tesla_radar_x159_id % 16;
        send_fake_message(RIR,RDTR,8,0x159,tesla_radar_can,MLB,MHB);
        //send 149
        MLB = 0x6A022600;
        MHB = 0x000F04AA + (tesla_radar_x149_id << 20);
        cksm = add_tesla_cksm2(MLB, MHB, 0x46, 7);
        MHB = MHB +(cksm << 24);
        tesla_radar_x149_id++;
        tesla_radar_x149_id = tesla_radar_x149_id % 16;
        send_fake_message(RIR,RDTR,8,0x149,tesla_radar_can,MLB,MHB);
        //send 129
        MLB = 0x20000000; 
        MHB = 0x00 + (tesla_radar_x129_id << 4);
        cksm = add_tesla_cksm2(MLB, MHB, 0x16, 5);
        MHB = MHB +(cksm << 8);
        tesla_radar_x129_id++;
        tesla_radar_x129_id = tesla_radar_x129_id % 16;
        send_fake_message(RIR,RDTR,6,0x129,tesla_radar_can,MLB,MHB);
        //send 1A9
        MLB = 0x000C0000 + (tesla_radar_x1A9_id << 28);
        MHB = 0x00;
        cksm = add_tesla_cksm2(MLB, MHB, 0x38, 4);
        MHB = MHB +(cksm << 8);
        tesla_radar_x1A9_id++;
        tesla_radar_x1A9_id = tesla_radar_x1A9_id % 16;
        send_fake_message(RIR,RDTR,5,0x1A9,tesla_radar_can,MLB,MHB);
    }
    //send all messages at 10Hz
    if (tesla_radar_counter % 10 ==0) {
        //send 209
        MLB = 0x5294FF00;
        MHB = 0x00800313;
        send_fake_message(RIR,RDTR,8,0x209,tesla_radar_can,MLB,MHB);
        //send 219
        MLB = 0x00000000; 
        MHB = 0x00000000;
        MHB = MHB + (tesla_radar_x219_id << 20);
        crc = add_tesla_crc(MLB, MHB,7);
        MHB = MHB +(crc << 24);
        tesla_radar_x219_id++;
        tesla_radar_x219_id = tesla_radar_x219_id % 16;
        send_fake_message(RIR,RDTR,8,0x219,tesla_radar_can,MLB,MHB);
    }
    //send all messages at 4Hz
    if (tesla_radar_counter % 25 ==0) {
        //send 2B9
        tesla_radar_x2B9_id = 0;
        int rec = 0x10 + tesla_radar_x2B9_id;
        if (rec == 0x10) {
            MLB = 0x00000000 | rec;
            MHB = radar_VIN_char(0,1) | radar_VIN_char(1,2) | radar_VIN_char(2,3);
        }
        if (rec == 0x11) {
            MLB = rec | radar_VIN_char(3,1) | radar_VIN_char(4,2) | radar_VIN_char(5,3);
            MHB = radar_VIN_char(6,0) | radar_VIN_char(7,1) | radar_VIN_char(8,2) | radar_VIN_char(9,3);
        }
        if (rec == 0x12) {
            MLB = rec | radar_VIN_char(10,1) | radar_VIN_char(11,2) | radar_VIN_char(12,3);
            MHB = radar_VIN_char(13,0) | radar_VIN_char(14,1) | radar_VIN_char(15,2) | radar_VIN_char(16,3);
        }
        tesla_radar_x2B9_id++;
        tesla_radar_x2B9_id = tesla_radar_x2B9_id % 3;
        send_fake_message(RIR,RDTR,8,0x2B9,tesla_radar_can,MLB,MHB);
    }
    //send all messages at 1Hz
    if (tesla_radar_counter ==0) {
        //send 2A9
        MLB = 0x41431642;
        MHB = 0x10020000;
        if ((sizeof(radar_VIN) >= 4) && ((int)(radar_VIN[7]) == 0x32)) {
            //also change to AWD if needed (most likely) if manual VIN and if position 8 of VIN is a 2 (dual motor)
            MLB = MLB | 0x08;
        }
        send_fake_message(RIR,RDTR,8,0x2A9,tesla_radar_can,MLB,MHB);
        //send 2D9
        MLB = 0x00834080;
        MHB = 0x00000000;
        send_fake_message(RIR,RDTR,8,0x2D9,tesla_radar_can,MLB,MHB);
    }
    tesla_radar_counter++;
    tesla_radar_counter = tesla_radar_counter % 100;
}

static void teslaradar_rx_hook(CAN_FIFOMailBox_TypeDef *to_push)
{
   int bus_number = (to_push->RDTR >> 4) & 0xFF;
  uint32_t addr;

  if (to_push->RIR & 4)
  {
    // Extended
    // Not looked at, but have to be separated
    // to avoid address collision
    addr = to_push->RIR >> 3;
  }
  else
  {
    // Normal
    addr = to_push->RIR >> 21;
  }

  if ((addr == tesla_radar_trigger_message_id) && (tesla_radar_trigger_message_id > 0)) {
    activate_tesla_radar(to_push->RIR,to_push->RDTR);
    return;
  }

  //looking for radar messages;
  if ((addr == 0x300) && (bus_number ==tesla_radar_can)) 
  {
    uint32_t ts = TIM2->CNT;
    uint32_t ts_elapsed = get_ts_elapsed(ts, tesla_last_radar_signal);
    if (tesla_radar_status == 1) {
      tesla_radar_status = 2;
      puts("Tesla Radar Active! \n");
      tesla_last_radar_signal = ts;
    } else
    if ((ts_elapsed > TESLA_RADAR_TIMEOUT) && (tesla_radar_status > 0)) {
      tesla_radar_status = 0;
      puts("Tesla Radar Inactive! (timeout 1) \n");
    } else 
    if ((ts_elapsed <= TESLA_RADAR_TIMEOUT) && (tesla_radar_status == 2)) {
      tesla_last_radar_signal = ts;
    }
    return;
  }

  //0x631 is sent by radar to initiate the sync
  if ((addr == 0x631) && (bus_number == 1))
  {
    uint32_t ts = TIM2->CNT;
    uint32_t ts_elapsed = get_ts_elapsed(ts, tesla_last_radar_signal);
    if (tesla_radar_status == 0) {
      tesla_radar_status = 1;
      tesla_last_radar_signal = ts;
      puts("Tesla Radar Initializing... \n");
    } else
    if ((ts_elapsed > TESLA_RADAR_TIMEOUT) && (tesla_radar_status > 0)) {
      tesla_radar_status = 0;
      puts("Tesla Radar Inactive! (timeout 2) \n");
    } else 
    if ((ts_elapsed <= TESLA_RADAR_TIMEOUT) && (tesla_radar_status > 0)) {
      tesla_last_radar_signal = ts;
    }
    return;
  }
}
