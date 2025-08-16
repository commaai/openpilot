#include <stdbool.h>

#include "opendbc/safety/board/fake_stm.h"
#include "opendbc/safety/board/can.h"

//int safety_tx_hook(CANPacket_t *msg) { return 1; }

#include "opendbc/safety/board/faults.h"
#include "opendbc/safety/safety.h"
#include "opendbc/safety/board/drivers/can_common.h"

// libsafety stuff
#include "opendbc/safety/tests/libsafety/safety_helpers.h"
