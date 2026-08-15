#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"

INCLUDE_BIG_MODEL=1 RELEASE_BRANCH=release-chestnut exec "$DIR/build_release.sh"
