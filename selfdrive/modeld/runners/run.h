#pragma once

#include "selfdrive/modeld/runners/runmodel.h"
#include "selfdrive/modeld/runners/snpemodel.h"

#if defined(USE_THNEED)
#include "selfdrive/modeld/runners/thneedmodel.h"
#elif defined(USE_ONNX_MODEL)
#include "selfdrive/modeld/runners/onnxmodel.h"
#endif
