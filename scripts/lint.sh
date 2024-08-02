#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd $DIR/../

# TODO: bring back rest of pre-commit checks:
# https://github.com/commaai/openpilot/blob/4b11c9e914707df9def598616995be2a5d355a6a/.pre-commit-config.yaml#L2

ruff check .
