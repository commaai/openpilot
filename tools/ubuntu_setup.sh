#!/usr/bin/env bash

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

# NOTE: this is used in a docker build, so do not run any scripts here.

# Optimized parallel setup for CI speed
echo "Starting optimized Ubuntu setup..."

# Run dependency installations in parallel for speed
"$DIR"/install_ubuntu_dependencies.sh &
UBUNTU_PID=$!

"$DIR"/install_python_dependencies.sh &
PYTHON_PID=$!

# Wait for both to complete
wait $UBUNTU_PID
wait $PYTHON_PID

echo "Ubuntu setup completed successfully"
