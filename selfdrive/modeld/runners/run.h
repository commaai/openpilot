#ifndef RUN_H
#define RUN_H

#include "runmodel.h"
#include "snpemodel.h"

#ifdef QCOM
  #define DefaultRunModel SNPEModel
#else
  #define DefaultRunModel SNPEModel
#endif

#endif
