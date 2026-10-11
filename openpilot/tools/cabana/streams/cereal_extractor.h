#pragma once

#include <capnp/message.h>
#include "tools/cabana/streams/cereal_series.h"
#include "tools/replay/logreader.h"

// Walks a non-CAN cereal Event using capnp dynamic reflection and extracts all
// numeric leaf values into `out` under paths like "/carState/vEgo".
void extractCerealEvent(const Event &e, double t, CerealSeriesMap &out);

// Overload for live streams where we have raw capnp data without an Event wrapper.
void extractCerealEvent(kj::ArrayPtr<const capnp::word> data, double t, CerealSeriesMap &out);
