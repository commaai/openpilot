#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

CLI="${TRANSLATOR_CLI:-}"
if [[ -z "$CLI" ]]; then
  command -v codex >/dev/null && CLI=codex || command -v claude >/dev/null && CLI=claude || {
    echo "Neither codex nor claude found in PATH."
    [[ -t 0 ]] && read -rp "Install one, then press Enter to exit... " _
    exit 1
  }
fi

PROMPT="Update openpilot UI translations in selfdrive/ui/translations using args: $*. Treat -a/-f/-t like old auto_translate. Translate English UI text naturally, preserve placeholders (%n, %1, {}, {:.1f}), HTML/tags, and plural forms, edit .po files in place, and print a short summary of changes."
if [[ "$CLI" == "codex" ]]; then
  exec codex exec --cd "$ROOT" --sandbox workspace-write "$PROMPT"
fi
cd "$ROOT"
exec claude -p --permission-mode bypassPermissions "$PROMPT"
