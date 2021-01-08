#ifndef RUN_H
#define RUN_H

#include "runmodel.h"
#include "snpemodel.h"

#if defined(QCOM) || defined(QCOM2)
  #include "thneedmodel.h"
#endif

#ifdef QCOM
  #define DefaultRunModel SNPEModel
#else
  #ifdef USE_ONNX_MODEL
    #include "onnxmodel.h"
    #define DefaultRunModel ONNXModel
  #else
    #define DefaultRunModel SNPEModel
  #endif
#endif

#endif
