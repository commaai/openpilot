#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
ROOT=$DIR/../
cd $ROOT

UPGRADE_FLAG=$([ "$1" != "--no-update" ] && echo "--upgrade" || : )
uv pip compile --preview --all-extras $UPGRADE_FLAG pyproject.toml -o requirements.txt
