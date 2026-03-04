#!/usr/bin/env bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
ROOT="$DIR/.."

cd "$ROOT"

git ls-files --recurse-submodules | grep -vE '\.git/|\.github/workflows/|matlab\..*\.md|\.lfsconfig|\.gitattributes|\.git$|\.gitmodules'
