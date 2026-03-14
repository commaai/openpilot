#!/bin/bash
set -e

readonly cur_script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

source $cur_script_dir/.venv/bin/activate
bash $cur_script_dir/tools/sim/launch_openpilot.sh
