#ifndef RUN_H
#define RUN_H

#include "runmodel.h"
#include "snpemodel.h"

#ifdef QCOM
  #define DefaultRunModel SNPEModel
#else
#define DefaultRunModel SNPEModel
  /* #include "tfmodel.h" */
  /* #define DefaultRunModel TFModel */
#endif

#endif
