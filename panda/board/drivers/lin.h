//--------------------------------------------------------------
// LIN Defines
//--------------------------------------------------------------
#define  LIN_SYNC_DATA               0x55  // SyncField (do not change)
#define  LIN_MAX_DATA                   8  // max 8 databytes


//--------------------------------------------------------------
// LIN error messages
//--------------------------------------------------------------
typedef enum {
  LIN_OK  = 0,   // no error
  LIN_WRONG_LEN, // wrong number of data
  LIN_RX_EMPTY,  // no frame received
  LIN_WRONG_CRC  // Checksum wrong
}LIN_ERR_t;

//--------------------------------------------------------------
// LIN frame Struct
//--------------------------------------------------------------
typedef struct {
  uint8_t has_response;          // set to 1 if message expects a response; else set to 0
  uint8_t frame_id;              // ID number of the frame
  uint8_t data_len;              // number of data bytes
  uint8_t data[LIN_MAX_DATA];    // data
}LIN_FRAME_t;

uint8_t p_LIN_makeChecksum(LIN_FRAME_t *frame);
void USART_SendBreak(uart_ring *u);


// --------------------------------------------------------------
// sends data via LIN interface
// frame:
// frame_id = Unique ID [0x00 to 0xFF]
// data_len = length of data to be sent
// data [] = data to be sent
//
// return_value:
// LIN_OK = Frame has been sent
// LIN_WRONG_LEN = invalid data length
// --------------------------------------------------------------
LIN_ERR_t LIN_SendData(uart_ring *LIN_UART, LIN_FRAME_t *frame)
{
  uint8_t checksum,n;

  // check the length
  if((frame -> data_len < 1) || (frame -> data_len > LIN_MAX_DATA)) {
    return(LIN_WRONG_LEN);
  }   

  // calculate checksum
  checksum=p_LIN_makeChecksum(frame);

  //------------------------
  // Break-Field
  //------------------------
  USART_SendBreak(LIN_UART);

  //------------------------
  // Sync-Field
  //------------------------
  putc(LIN_UART, LIN_SYNC_DATA);

  //------------------------
  // ID-Field
  //------------------------
  putc(LIN_UART, frame->frame_id);
  
  //------------------------
  // Data-Field [1...n]
  //------------------------  
  for(n=0; n < frame -> data_len; n++) {
    putc(LIN_UART, frame -> data[n]);
  }

  //------------------------
  // CRC-Field
  //------------------------
  putc(LIN_UART, checksum);

  return(LIN_OK);    
}


// --------------------------------------------------------------
// receives data via LIN interface
// frame:
// frame_id = Unique ID [0x00 to 0xFF]
// data_len = number of data to be received
// return:
// data [] = data received (if LIN_OK)
//
// return_value:
// LIN_OK = Frame was received
// LIN_WRONG_LEN = wrong number of data
// LIN_RX_EMPTY = no frame received
// LIN_WRONG_CRC = Checksum wrong
// --------------------------------------------------------------
LIN_ERR_t LIN_SendReceiveFrame(uart_ring *LIN_UART, LIN_FRAME_t *frame)
{
  // check the length
  if((frame -> data_len < 1) || (frame -> data_len > LIN_MAX_DATA)) {
    return(LIN_WRONG_LEN);
  }
  
  //-------------------------------
  // Break-Field
  //-------------------------------
  USART_SendBreak(LIN_UART);

  //-------------------------------
  // Sync-Field
  //-------------------------------
  putc(LIN_UART, LIN_SYNC_DATA);

  //-------------------------------
  // ID-Field
  //-------------------------------
  putc(LIN_UART, frame->frame_id);

  //now wait for the slave device to respond with the data

  return(LIN_OK);
}

LIN_ERR_t LIN_ReceiveData(uart_ring *LIN_UART, LIN_FRAME_t *frame)
{
  //-------------------------------
  // copy received data
  //-------------------------------
  
  //TODO: Get receive working. Do I need to add a small delay here?
  
  uint8_t *resp = 0;
  int resp_len = 0;
  
  while ((resp_len < frame -> data_len) && getc(LIN_UART, (char*)&resp[resp_len])) {
    ++resp_len;
    frame->data[resp_len]=resp;
  }
  
  return(LIN_OK);
}

void USART_SendBreak(uart_ring *u)
{
    SET_BIT(u -> uart -> CR1, USART_CR1_SBK);
}

// --------------------------------------------------------------
// internal function
// Calculate checksum over all data
// (classic-mode = inverted modulo256 sum)
//
// ret_value = checksum
//--------------------------------------------------------------
uint8_t p_LIN_makeChecksum(LIN_FRAME_t *frame)
{
  uint8_t ret_value=0, n;
  uint16_t dummy;

  // calculate checksum  
  dummy = 0;
  for(n = 0; n < frame -> data_len; n++) {
    dummy += frame -> data[n];
    if (dummy > 0xFF) {
      dummy -= 0xFF;
    } 
  }
  ret_value = (uint8_t)(dummy);
  ret_value ^= 0xFF;
  
  return(ret_value);
}
