#include "common_dbc.h"

namespace {

const Signal sigs_368[] = {
    {
      .name = "FrictionBrakePressure",
      .b1 = 16,
      .b2 = 16,
      .bo = 32,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_560[] = {
    {
      .name = "Regen",
      .b1 = 6,
      .b2 = 10,
      .bo = 48,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_789[] = {
    {
      .name = "FrictionBrakeCmd",
      .b1 = 4,
      .b2 = 12,
      .bo = 48,
      .is_signed = true,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "FrictionBrakeMode",
      .b1 = 0,
      .b2 = 4,
      .bo = 60,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "FrictionBrakeChecksum",
      .b1 = 16,
      .b2 = 16,
      .bo = 32,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "RollingCounter",
      .b1 = 38,
      .b2 = 2,
      .bo = 24,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_823[] = {
    {
      .name = "SteeringWheelCmd",
      .b1 = 16,
      .b2 = 16,
      .bo = 32,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "RollingCounter",
      .b1 = 36,
      .b2 = 2,
      .bo = 26,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "SteeringWheelChecksum",
      .b1 = 40,
      .b2 = 16,
      .bo = 8,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};

const Msg msgs[] = {
  {
    .name = "EBCMFrictionBrakeStatus",
    .address = 0x170,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_368),
    .sigs = sigs_368,
  },
  {
    .name = "EBCMRegen",
    .address = 0x230,
    .size = 6,
    .num_sigs = ARRAYSIZE(sigs_560),
    .sigs = sigs_560,
  },
  {
    .name = "EBCMFrictionBrakeCmd",
    .address = 0x315,
    .size = 5,
    .num_sigs = ARRAYSIZE(sigs_789),
    .sigs = sigs_789,
  },
  {
    .name = "PACMParkAssitCmd",
    .address = 0x337,
    .size = 7,
    .num_sigs = ARRAYSIZE(sigs_823),
    .sigs = sigs_823,
  },
};

const Val vals[] = {
};

}

const DBC gm_global_a_chassis = {
  .name = "gm_global_a_chassis",
  .num_msgs = ARRAYSIZE(msgs),
  .msgs = msgs,
  .vals = vals,
  .num_vals = ARRAYSIZE(vals),
};

dbc_init(gm_global_a_chassis)