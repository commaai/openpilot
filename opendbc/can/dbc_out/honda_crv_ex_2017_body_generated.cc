#include "common_dbc.h"

namespace {

const Signal sigs_318291615[] = {
    {
      .name = "BSM_ALERT",
      .b1 = 3,
      .b2 = 1,
      .bo = 60,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "BSM_MODE",
      .b1 = 1,
      .b2 = 2,
      .bo = 61,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_318291879[] = {
    {
      .name = "BSM_ALERT",
      .b1 = 3,
      .b2 = 1,
      .bo = 60,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "BSM_MODE",
      .b1 = 1,
      .b2 = 2,
      .bo = 61,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};

const Msg msgs[] = {
  {
    .name = "BSM_STATUS_LEFT",
    .address = 0x12F8BE9F,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_318291615),
    .sigs = sigs_318291615,
  },
  {
    .name = "BSM_STATUS_RIGHT",
    .address = 0x12F8BFA7,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_318291879),
    .sigs = sigs_318291879,
  },
};

const Val vals[] = {
    {
      .name = "BSM_MODE",
      .address = 0x12F8BE9F,
      .def_val = "2 BLIND_SPOT 1 CROSS_TRAFFIC 0 OFF",
      .sigs = sigs_318291615,
    },
    {
      .name = "BSM_MODE",
      .address = 0x12F8BFA7,
      .def_val = "2 BLIND_SPOT 1 CROSS_TRAFFIC 0 OFF",
      .sigs = sigs_318291879,
    },
};

}

const DBC honda_crv_ex_2017_body_generated = {
  .name = "honda_crv_ex_2017_body_generated",
  .num_msgs = ARRAYSIZE(msgs),
  .msgs = msgs,
  .vals = vals,
  .num_vals = ARRAYSIZE(vals),
};

dbc_init(honda_crv_ex_2017_body_generated)