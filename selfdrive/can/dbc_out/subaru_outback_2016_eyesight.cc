#include <cstdint>

#include "common.h"

namespace {

const Signal sigs_2[] = {
    {
      .name = "SteeringAngle",
      .b1 = 0,
      .b2 = 16,
      .bo = 48,
      .is_signed = true,
      .factor = 0.1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_209[] = {
    {
      .name = "BrakePosition",
      .b1 = 16,
      .b2 = 8,
      .bo = 40,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_320[] = {
    {
      .name = "ThrottlePosition",
      .b1 = 0,
      .b2 = 8,
      .bo = 56,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_324[] = {
    {
      .name = "CruiseButtons",
      .b1 = 3,
      .b2 = 2,
      .bo = 59,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "BrakeApplied",
      .b1 = 15,
      .b2 = 1,
      .bo = 48,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "CruiseEnabled",
      .b1 = 55,
      .b2 = 1,
      .bo = 8,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "BrakeStatus",
      .b1 = 52,
      .b2 = 1,
      .bo = 11,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_642[] = {
    {
      .name = "TurnSignal",
      .b1 = 42,
      .b2 = 2,
      .bo = 20,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_884[] = {
    {
      .name = "DoorOpenFD",
      .b1 = 31,
      .b2 = 1,
      .bo = 32,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "DoorOpenFP",
      .b1 = 30,
      .b2 = 1,
      .bo = 33,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "DoorOpenRP",
      .b1 = 29,
      .b2 = 1,
      .bo = 34,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "DoorOpenRD",
      .b1 = 28,
      .b2 = 1,
      .bo = 35,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "DoorOpenHatch",
      .b1 = 27,
      .b2 = 1,
      .bo = 36,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};

const Msg msgs[] = {
  {
    .name = "Steering",
    .address = 0x2,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_2),
    .sigs = sigs_2,
  },
  {
    .name = "NEW_MSG_1",
    .address = 0xD1,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_209),
    .sigs = sigs_209,
  },
  {
    .name = "Throttle",
    .address = 0x140,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_320),
    .sigs = sigs_320,
  },
  {
    .name = "CruiseControl",
    .address = 0x144,
    .size = 7,
    .num_sigs = ARRAYSIZE(sigs_324),
    .sigs = sigs_324,
  },
  {
    .name = "NEW_MSG_2",
    .address = 0x282,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_642),
    .sigs = sigs_642,
  },
  {
    .name = "DoorStatus",
    .address = 0x374,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_884),
    .sigs = sigs_884,
  },
};

const Val vals[] = {
    {
      .name = "BrakeStatus",
      .address = 0x144,
      .def_val = "1 ON 0 OFF",
      .sigs = sigs_324,
    },
    {
      .name = "BrakeApplied",
      .address = 0x144,
      .def_val = "1 ON 0 OFF",
      .sigs = sigs_324,
    },
    {
      .name = "CruiseEnabled",
      .address = 0x144,
      .def_val = "1 ON 0 OFF",
      .sigs = sigs_324,
    },
    {
      .name = "CruiseButtons",
      .address = 0x144,
      .def_val = "2 SET 1 RESUME",
      .sigs = sigs_324,
    },
    {
      .name = "TurnSignal",
      .address = 0x282,
      .def_val = "2 LEFT 1 RIGHT",
      .sigs = sigs_642,
    },
};

}

const DBC subaru_outback_2016_eyesight = {
  .name = "subaru_outback_2016_eyesight",
  .num_msgs = ARRAYSIZE(msgs),
  .msgs = msgs,
  .vals = vals,
  .num_vals = ARRAYSIZE(vals),
};

dbc_init(subaru_outback_2016_eyesight)