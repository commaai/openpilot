#pragma once

#ifdef QCOM
#include "selfdrive/hardware/eon/hardware.h"
Hardware
#elif QCOM2
#include "selfdrive/hardware/tici/hardware.h"
#else
#include "selfdrive/hardware/pc/hardware.h"
#endif

static inline Hardware HARDWARE;
