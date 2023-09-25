#!/bin/bash
set -e

echo "WARNING: this is a script for internal use only."

git lfs push --all ssh://git@gitlab.com/commaai/openpilot-lfs.git
