#pragma once

#ifdef QCOM
#include "selfdrive/hardware/eon/hardware.h"
#elif QCOM2
#include "selfdrive/hardware/tici/hardware.h"
#else
#include "selfdrive/hardware/pc/hardware.h"
#endif

static inline const Hardware HARDWARE;
