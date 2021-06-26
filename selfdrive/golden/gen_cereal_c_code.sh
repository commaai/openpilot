#!/usr/bin/bash

cd /data/openpilot/cereal/
mkdir -p /data/openpilot/cereal/gen/c/
capnpc car.capnp log.capnp --src-prefix=. -o c:/data/openpilot/cereal/gen/c/