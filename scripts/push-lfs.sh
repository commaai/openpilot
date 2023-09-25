#!/bin/bash
set -e

printf "WARNING: this is a script for internal use only.\n\n"

printf "pushing to GitLab\n"
git lfs push --all ssh://git@gitlab.com/commaai/openpilot-lfs.git

printf "\npushing to GitHub\n"
git lfs push --all git@github.com:commaai/openpilot.git
