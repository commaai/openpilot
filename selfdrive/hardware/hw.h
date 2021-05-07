#pragma once

#include "selfdrive/hardware/base.h"

#ifdef QCOM
#include "selfdrive/hardware/eon/hardware.h"
#define Hardware HardwareEon
#elif QCOM2
#include "selfdrive/hardware/tici/hardware.h"
#define Hardware HardwareTici
#else
#include "selfdrive/hardware/pc/hardware.h"
#define Hardware HardwarePC
#endif
