typedef struct {
  uint8_t ptr;
  uint8_t tail_size;
  uint8_t data[72];
  uint8_t counter;
} usb_asm_buffer;

usb_asm_buffer ep1_buffer = {.ptr = 0, .tail_size = 0, .counter = 0};
uint32_t total_rx_size = 0;

int usb_cb_ep1_in(void *usbdata, int len, bool hardwired) {
  UNUSED(hardwired);
  uint8_t pos = 1;
  uint8_t *usbdata8 = (uint8_t *)usbdata;
  usbdata8[0] = ep1_buffer.counter;
  // Send tail of previous message if it is in buffer
  if (ep1_buffer.ptr > 0) {
    if (ep1_buffer.ptr <= 63U) {
      (void)memcpy(&usbdata8[pos], ep1_buffer.data, ep1_buffer.ptr);
      pos += ep1_buffer.ptr;
      ep1_buffer.ptr = 0;
    } else {
      (void)memcpy(&usbdata8[pos], ep1_buffer.data, 63U);
      ep1_buffer.ptr = ep1_buffer.ptr - 63U;
      (void)memcpy(ep1_buffer.data, &ep1_buffer.data[63U], ep1_buffer.ptr);
      pos += 63U;
    }
  }

  if (total_rx_size > MAX_EP1_CHUNK_PER_BULK_TRANSFER) {
    total_rx_size = 0;
    ep1_buffer.counter = 0;
  } else {
    CANPacket_t can_packet;
    while ((pos < len) && can_pop(&can_rx_q, &can_packet)) {
      uint8_t pckt_len = CANPACKET_HEAD_SIZE + dlc_to_len[can_packet.data_len_code];
      if ((pos + pckt_len) <= len) {
        (void)memcpy(&usbdata8[pos], &can_packet, pckt_len);
        pos += pckt_len;
      } else {
        (void)memcpy(&usbdata8[pos], &can_packet, len - pos);
        ep1_buffer.ptr = pckt_len - (len - pos);
        //(void)memcpy(ep1_buffer.data, ((uint8_t*)&can_packet + (len - pos)), ep1_buffer.ptr);
        // cppcheck-suppress objectIndex
        (void)memcpy(ep1_buffer.data, &((uint8_t*)&can_packet)[(len - pos)], ep1_buffer.ptr);
        pos = len;
      }
    }
    ep1_buffer.counter++;
    total_rx_size += pos;
  }
  if (pos != len) {
    ep1_buffer.counter = 0;
    total_rx_size = 0;
  }
  if (pos <= 1) { pos = 0; }
  return pos;
}

usb_asm_buffer ep3_buffer = {.ptr = 0, .tail_size = 0, .counter = 0};

// send on CAN
void usb_cb_ep3_out(void *usbdata, int len, bool hardwired) {
  UNUSED(hardwired);
  uint8_t *usbdata8 = (uint8_t *)usbdata;
  // Got first packet from a stream, resetting buffer and counter
  if (usbdata8[0] == 0) {
    ep3_buffer.counter = 0;
    ep3_buffer.ptr = 0;
    ep3_buffer.tail_size = 0;
  }
  // Assembling can message with data from buffer
  if (usbdata8[0] == ep3_buffer.counter) {
    uint8_t pos = 1;
    ep3_buffer.counter++;
    if (ep3_buffer.ptr != 0) {
      if (ep3_buffer.tail_size <= 63U) {
        CANPacket_t to_push;
        (void)memcpy(&ep3_buffer.data[ep3_buffer.ptr], &usbdata8[pos], ep3_buffer.tail_size);
        (void)memcpy(&to_push, ep3_buffer.data, ep3_buffer.ptr + ep3_buffer.tail_size);
        can_send(&to_push, to_push.bus, false);
        pos += ep3_buffer.tail_size;
        ep3_buffer.ptr = 0;
        ep3_buffer.tail_size = 0;
      } else {
        (void)memcpy(&ep3_buffer.data[ep3_buffer.ptr], &usbdata8[pos], len - pos);
        ep3_buffer.tail_size -= 63U;
        ep3_buffer.ptr += 63U;
        pos += 63U;
      }
    }

    while (pos < len) {
      uint8_t pckt_len = CANPACKET_HEAD_SIZE + dlc_to_len[(usbdata8[pos] >> 4U)];
      if ((pos + pckt_len) <= (uint8_t)len) {
        CANPacket_t to_push;
        (void)memcpy(&to_push, &usbdata8[pos], pckt_len);
        can_send(&to_push, to_push.bus, false);
        pos += pckt_len;
      } else {
        (void)memcpy(ep3_buffer.data, &usbdata8[pos], len - pos);
        ep3_buffer.ptr = len - pos;
        ep3_buffer.tail_size = pckt_len - ep3_buffer.ptr;
        pos += ep3_buffer.ptr;
      }
    }
  }
}
