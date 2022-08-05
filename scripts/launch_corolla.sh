#!/usr/bin/bash

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"

export FINGERPRINT="TOYOTA COROLLA TSS2 2019"
$DIR/../launch_openpilot.sh
