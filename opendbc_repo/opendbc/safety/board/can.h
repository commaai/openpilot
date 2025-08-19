#pragma once

// TODO: clean this up. it's for interop with the panda version
#ifndef CANPACKET_HEAD_SIZE

#include "opendbc/safety/board/can_declarations.h"

static const unsigned char dlc_to_len[] = {0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 12U, 16U, 20U, 24U, 32U, 48U, 64U};

#endif
