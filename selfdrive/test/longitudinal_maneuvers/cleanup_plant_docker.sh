#!/bin/sh
docker run -v $(pwd)/docker_out:/tmp/openpilot/selfdrive/test/out openpilot_ci /bin/bash -c "rm -Rf /tmp/openpilot/selfdrive/test/out/longitudinal"
