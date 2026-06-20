#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd)"
ROOT="$DIR/../../../"

cd $DIR
./update_translations.py

command -v codex >/dev/null || {
  echo "Install codex CLI to continue:"
  echo "-> https://developers.openai.com/codex/cli"
  echo
  exit 1
}

codex exec --cd "$ROOT" -c 'model_reasoning_effort="low"' --dangerously-bypass-approvals-and-sandbox "$(cat <<EOF
Update openpilot UI translations in selfdrive/ui/translations.
- Translate English UI text naturally.
- Preserve placeholders (%n, %1, {}, {:.1f}), HTML/tags, and plural forms.
- Edit .po files in place.
- Print a short summary of changes.
- All strings should be translated. Don't stop until it's 100%.
- Be mindful of the layout/style of the UI and length of the original English string.
EOF
)"
