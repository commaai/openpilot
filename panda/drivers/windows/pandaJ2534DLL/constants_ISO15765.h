#pragma once

#define msg_is_extaddr(msg) check_bmask(msg->TxFlags, ISO15765_ADDR_TYPE)
#define msg_is_padded(msg) check_bmask(msg->TxFlags, ISO15765_FRAME_PAD)

#define FRAME_SINGLE   0x00
#define FRAME_FIRST    0x10
#define FRAME_CONSEC   0x20
#define FRAME_FLOWCTRL 0x30

#define FLOWCTRL_CONTINUE 0
#define FLOWCTRL_WAIT     1
#define FLOWCTRL_ABORT    2

#define msg_get_type(msg, addrlen)   ((msg).Data[addrlen] & 0xF0)

#define is_single(msg, addrlen)      (msg_get_type(msg, addrlen) == FRAME_SINGLE)
#define is_first(msg, addrlen)       (msg_get_type(msg, addrlen) == FRAME_FIRST)
#define is_consecutive(msg, addrlen) (msg_get_type(msg, addrlen) == FRAME_CONSEC)
#define is_flowctrl(msg, addrlen)    (msg_get_type(msg, addrlen) == FRAME_FLOWCTRL)