#!/usr/bin/env bash
set -euo pipefail

# Cross-distro smoketest — runs openpilot setup+build+lint+test in Docker

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$DIR/.." && pwd)"
COMPOSE_FILE="$(mktemp /tmp/smoketest-compose.XXXXXX.yml)"

cat > "$COMPOSE_FILE" <<YAML
x-smoketest: &smoketest
  volumes:
    - $ROOT:/src:ro
  dns: 8.8.8.8
  environment:
    CI: "1"
    LANG: en_US.UTF-8
    LC_ALL: en_US.UTF-8
    LIBGL_ALWAYS_SOFTWARE: "1"
    SCALE: "1"
  entrypoint: ["/bin/bash", "-exc"]
  command:
    - |
      cp -a /src /tmp/openpilot
      cd /tmp/openpilot
      printf '[safe]\n\tdirectory = *\n' > \$\$HOME/.gitconfig
      tools/op.sh setup
      source .venv/bin/activate
      export DISPLAY=:99
      Xvfb :99 -screen 0 1024x768x24 +extension GLX &
      scons
      scripts/lint/lint.sh
      MAX_EXAMPLES=1 pytest -x -m 'not slow' --deselect tools/replay/tests/test_replay

services:
  ubuntu-20.04:
    <<: *smoketest
    image: ubuntu:20.04
  ubuntu-24.04:
    <<: *smoketest
    image: ubuntu:24.04
  debian-12:
    <<: *smoketest
    image: debian:13
  fedora-41:
    <<: *smoketest
    image: fedora:41
  arch:
    <<: *smoketest
    image: archlinux:latest
YAML

COMPOSE="docker compose -f $COMPOSE_FILE"

# Clean stale containers, then run
$COMPOSE down --remove-orphans 2>/dev/null || true
$COMPOSE up --force-recreate "$@" || true

echo ""
echo "══════════════════════════════════════"
echo "  Smoketest Results"
echo "══════════════════════════════════════"
$COMPOSE ps -a "$@" --format "table {{.Service}}\t{{.State}}"

# Exit non-zero if any container failed
! $COMPOSE ps -a "$@" --format "{{.ExitCode}}" | grep -qv '^0$'
