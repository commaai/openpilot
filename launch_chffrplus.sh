#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

# we can delete this after most users have the new launch_openpilot.sh

exec "$DIR/launch_openpilot.sh" "$@"
