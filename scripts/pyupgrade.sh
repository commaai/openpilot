#!/usr/bin/env bash

set -e

pip install --upgrade pyupgrade

git ls-files '*.py' | grep -v 'third_party/' | xargs pyupgrade --py311-plus
